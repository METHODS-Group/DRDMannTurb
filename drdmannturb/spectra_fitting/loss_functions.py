"""Implements the loss function in an extensible and configurable manner without directly exposing the calculation."""

import torch
from torch.utils.tensorboard import SummaryWriter

from ..parameters import LossParameters


class LossAggregator:
    """Aggregator for all loss function terms."""

    def __init__(
        self,
        params: LossParameters,
        k1space: torch.Tensor,
        zref: float,
        tb_log_dir: str | None = None,
        tb_comment: str = "",
    ):
        r"""Initialize aggregator for given loss function parameters.

        The loss function for spectra fitting is determined by the following minimization problem:

        .. math::
            \min _{\boldsymbol{\theta}}\left\{\operatorname{MSE}[\boldsymbol{\theta}]+
            \alpha \operatorname{Pen}[\boldsymbol{\theta}]+
            \beta \operatorname{Reg}\left[\boldsymbol{\theta}_{\mathrm{NN}}\right]\right\}

        where the loss terms are defined in the class's methods. In the following, :math:`L` is the number of data
        points :math:`f_j \in \mathcal{D}`. The data :math:`J_i (f_j)` is evaluated using the Kaimal spectra. Note that
        the norm :math:`||\cdot||_{\mathcal{D}}` is defined as

        .. math::
            \|g\|_{\mathcal{D}}^2:=\int_{\mathcal{D}}|g(f)|^2 \mathrm{~d}(\log f).

        Additionally, writes results per epoch to TensorBoard.

        Parameters
        ----------
        params : LossParameters
            Dataclass with parameters that determine the loss function
        k1space : torch.Tensor
            The spectra space for k1, this is assumed to be in logspace.
        zref : float
            Reference velocity, needed for computing penalty derivatives wrt ``k1 z``.
        tb_log_dir : Optional[str]
            Logging directory for the TensorBoard logger. Conventions are those of TensorBoard, which by default result
            in the creation of a ``runs`` subdirectory where the script is being run if this parameter is left as None.
        fn_comment : str
            Filename comment used by tensorboard; useful for distinguishing between architectures and hyperparameters.
            Refer to tensorboard documentation for examples of use. By default, the empty string, which results in
            default tensorboard filenames.
        """
        self.writer = SummaryWriter(log_dir=tb_log_dir, comment=tb_comment)
        self.params = params

        self.zref = zref
        self.k1space = k1space
        self.logk1 = torch.log(self.k1space)
        self.h1 = torch.diff(self.logk1)
        self.h2 = torch.diff(0.5 * (self.logk1[:-1] + self.logk1[1:]))

        def t_alphapen1(y, theta_NN, epoch):
            return self.Pen1stOrder(y, epoch) if self.params.alpha_pen1 else 0.0

        def t_alphapen2(y, theta_NN, epoch):
            return self.Pen2ndOrder(y, epoch) if self.params.alpha_pen2 else 0.0

        def t_beta_reg(y, theta_NN, epoch):
            return self.Regularization(theta_NN, epoch) if self.params.beta_reg else 0.0

        self.loss_func = (
            lambda y, theta_NN, epoch: t_alphapen1(y, theta_NN, epoch)
            + t_alphapen2(y, theta_NN, epoch)
            + t_beta_reg(y, theta_NN, epoch)
        )

    def MSE_term(self, model: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
        r"""Evaluate the MSE loss term.

        This is given by

        .. math::
            \operatorname{MSE}[\boldsymbol{\theta}]:=\frac{1}{L} \sum_{i=1}^4
            \sum_{j=1}^L\left(\log \left|J_i\left(f_j\right)\right|-\log
            \left|\widetilde{J}_i\left(f_j, \boldsymbol{\theta}\right)\right|\right)^2,

        Parameters
        ----------
        model : torch.Tensor
            Model output of spectra on the k1space provided in the constructor.
        target : torch.Tensor
            True spectra data on the k1space provided in the constructor.
        epoch : int
            Epoch number, used for the TensorBoard loss writer.

        Returns
        -------
        torch.Tensor
            Evaluated MSE loss term.
        """
        mse_loss = torch.mean(torch.log(torch.abs(model / target)).square())

        self.writer.add_scalar("MSE Loss", mse_loss, epoch)

        return mse_loss

    def Pen2ndOrder(self, y: torch.Tensor, epoch: int) -> torch.Tensor:
        r"""Evaluate the second-order penalization term.

        This is given by

        .. math::
            \operatorname{Pen}_2[\boldsymbol{\theta}]:=\frac{1}{|\mathcal{D}|}
            \sum_{i=1}^4\left\|\operatorname{ReLU}\left(\frac{\partial^2 \log
            \left|\widetilde{J}_i(\cdot, \boldsymbol{\theta})\right|}{\left(
            \partial \log k_1\right)^2}\right)\right\|_{\mathcal{D}}^2,

        Parameters
        ----------
        y : torch.Tensor
            Model spectra output.
        epoch : int
            Epoch number, used for the TensorBoard loss writer.

        Returns
        -------
        torch.Tensor
            2nd order penalty loss term.
        """
        logy = torch.log(torch.abs(y))
        d2logy = torch.diff(torch.diff(logy, dim=-1) / self.h1, dim=-1) / self.h2

        pen2ndorder_loss = self.params.alpha_pen2 * torch.mean(torch.relu(d2logy).square()) / self.zref**2

        self.writer.add_scalar("2nd Order Penalty", pen2ndorder_loss, epoch)

        return pen2ndorder_loss

    def Pen1stOrder(self, y: torch.Tensor, epoch: int) -> torch.Tensor:
        r"""Evaluate the first-order penalization term.

        This is given by

        .. math::
            \operatorname{Pen}_1[\boldsymbol{\theta}]:=\frac{1}{|\mathcal{D}|}
            \sum_{i=1}^4\left\|\operatorname{ReLU}\left(\frac{\partial \log
            \left|\widetilde{J}_i(\cdot, \boldsymbol{\theta})\right|}{\partial
            \log k_1}\right)\right\|_{\mathcal{D}}^2.


        Parameters
        ----------
        y : torch.Tensor
            Model spectra output.
        epoch : int
            Epoch number, used for the TensorBoard loss writer.


        Returns
        -------
        torch.Tensor
            1st order penalty loss.
        """
        logy = torch.log(torch.abs(y))
        d1logy = torch.diff(logy, dim=-1) / self.h1

        pen1storder_loss = self.params.alpha_pen1 * torch.mean(torch.relu(d1logy).square()) / self.zref
        self.writer.add_scalar("1st Order Penalty", pen1storder_loss, epoch)

        return pen1storder_loss

    def Regularization(self, theta_NN: torch.Tensor, epoch: int) -> torch.Tensor:
        r"""Evaluate the regularization term.

        .. math::
            \operatorname{Reg}\left[\boldsymbol{\theta}_{\mathrm{NN}}\right]:=\frac{1}{N}
            \sum_{i=1}^N \theta_{\mathrm{NN}, i}^2.

        Parameters
        ----------
        theta_NN : torch.Tensor
            Neural network parameters.
        epoch : int
            Epoch number, used for the TensorBoard loss writer.

        Returns
        -------
        torch.Tensor
            Regularization loss of neural network model.
        """
        if theta_NN is None:  # TODO: Is this needed?
            raise ValueError(
                "Regularization loss requires a neural network model to be used. Set the regularization hyperparameter\
                (beta_reg) to 0 if using a non-neural network model."
            )

        reg_loss = self.params.beta_reg * theta_NN.square().mean()

        self.writer.add_scalar("Regularization", reg_loss, epoch)

        return reg_loss

    def eval(
        self,
        y: torch.Tensor,
        y_data: torch.Tensor,
        theta_NN: torch.Tensor | None,
        epoch: int,
    ) -> torch.Tensor:
        """Evaluate the full loss term at a given epoch.

        This method sequentially evaluates each term in the loss and returns the sum total loss.

        Parameters
        ----------
        y : torch.Tensor
            Model spectra output on k1space in constructor.
        y_data : torch.Tensor
            True spectra data on k1space in constructor.
        theta_NN : Optional[torch.Tensor]
            Neural network parameters, used in the regularization loss term if activated. Can be set to None if no
            neural network is used.
        epoch : int
            Epoch number, used for the TensorBoard loss writer.

        Returns
        -------
        torch.Tensor
            Evaluated total loss, as a weighted sum of all terms with non-zero hyperparameters.
        """
        total_loss = self.MSE_term(y, y_data, epoch) + self.loss_func(y, theta_NN, epoch)

        self.writer.add_scalar("Total Loss", total_loss, epoch)

        return total_loss
