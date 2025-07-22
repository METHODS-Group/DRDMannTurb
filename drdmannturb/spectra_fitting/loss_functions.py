"""Implements the loss function in an extensible and configurable manner without directly exposing the calculation."""

import torch
from torch.utils.tensorboard import SummaryWriter

from ..parameters import LossParameters


class LossAggregator:
    """Aggregator for all loss function terms with component tracking."""

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

        def t_alphapen1(y, theta_NN, epoch, **kwargs):
            return self.Pen1stOrder(y, epoch) if self.params.alpha_pen1 else 0.0

        def t_alphapen2(y, theta_NN, epoch, **kwargs):
            return self.Pen2ndOrder(y, epoch) if self.params.alpha_pen2 else 0.0

        def t_beta_reg(y, theta_NN, epoch, **kwargs):
            return self.Regularization(theta_NN, epoch) if self.params.beta_reg else 0.0

        def t_gamma_coherence(y, theta_NN, epoch, **kwargs):
            if self.params.gamma_coherence and "coherence_data" in kwargs:
                coh_data = kwargs["coherence_data"]
                return self.CoherenceLoss(
                    coh_data["model_u"],
                    coh_data["model_v"],
                    coh_data["model_w"],
                    coh_data["data_u"],
                    coh_data["data_v"],
                    coh_data["data_w"],
                    epoch,
                )
            return 0.0

        self.component_losses: dict[str, list[float]] = {
            "mse": [],
            "coherence": [],
            "pen1": [],
            "pen2": [],
            "regularization": [],
            "total": [],
        }

        self.running_averages: dict[str, float] = {
            "mse": 0.0,
            "coherence": 0.0,
            "pen1": 0.0,
            "pen2": 0.0,
            "regularization": 0.0,
        }

        self.auto_balance = getattr(params, "auto_balance_losses", False)
        self.balance_freq = 10
        self.loss_scales: dict[str, float] = {}

        self.ema_decay = 0.9

        self.loss_func = (
            lambda y, theta_NN, epoch, **kwargs: t_alphapen1(y, theta_NN, epoch, **kwargs)
            + t_alphapen2(y, theta_NN, epoch, **kwargs)
            + t_beta_reg(y, theta_NN, epoch, **kwargs)
            + t_gamma_coherence(y, theta_NN, epoch, **kwargs)
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

        if mse_loss.isnan():
            print("MSE loss is NaN")
            print(f"model: {model}")
            print(f"target: {target}")
            print(f"mse_loss: {mse_loss}")
            raise ValueError("MSE loss is NaN")

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

        if pen2ndorder_loss.isnan():
            print("2nd order penalty loss is NaN")
            print(f"y: {y}")
            print(f"logy: {logy}")
            print(f"d2logy: {d2logy}")
            print(f"pen2ndorder_loss: {pen2ndorder_loss}")
            raise ValueError("2nd order penalty loss is NaN")

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

        if pen1storder_loss.isnan():
            print("1st order penalty loss is NaN")
            print(f"y: {y}")
            print(f"logy: {logy}")
            print(f"d1logy: {d1logy}")
            print(f"pen1storder_loss: {pen1storder_loss}")
            raise ValueError("1st order penalty loss is NaN")

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
        reg_loss = self.params.beta_reg * theta_NN.square().mean()

        self.writer.add_scalar("Regularization", reg_loss, epoch)

        if reg_loss.isnan():
            print("Regularization loss is NaN")
            print(f"theta_NN: {theta_NN}")
            print(f"reg_loss: {reg_loss}")
            raise ValueError("Regularization loss is NaN")

        return reg_loss

    def CoherenceLoss(
        self,
        model_coherence_u: torch.Tensor,
        model_coherence_v: torch.Tensor,
        model_coherence_w: torch.Tensor,
        data_coherence_u: torch.Tensor,
        data_coherence_v: torch.Tensor,
        data_coherence_w: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        r"""Evaluate the coherence loss term.

        Computes MSE between model and experimental coherence functions:

        .. math::
            \text{CoherenceLoss} = \frac{1}{N} \sum_{i \in \{u,v,w\}} \sum_{r,f}
            \left( C_{ii}^{\text{model}}(r,f) - C_{ii}^{\text{data}}(r,f) \right)^2

        Parameters
        ----------
        model_coherence_u : torch.Tensor
            Model coherence for u-component. Shape: (n_separations, n_frequencies)
        model_coherence_v : torch.Tensor
            Model coherence for v-component. Shape: (n_separations, n_frequencies)
        model_coherence_w : torch.Tensor
            Model coherence for w-component. Shape: (n_separations, n_frequencies)
        data_coherence_u : torch.Tensor
            Experimental coherence for u-component. Shape: (n_separations, n_frequencies)
        data_coherence_v : torch.Tensor
            Experimental coherence for v-component. Shape: (n_separations, n_frequencies)
        data_coherence_w : torch.Tensor
            Experimental coherence for w-component. Shape: (n_separations, n_frequencies)
        epoch : int
            Epoch number, used for the TensorBoard loss writer.

        Returns
        -------
        torch.Tensor
            Evaluated coherence loss term.
        """
        # Convert data to tensors if needed and ensure they're on the same device
        if not torch.is_tensor(data_coherence_u):
            data_coherence_u = torch.tensor(
                data_coherence_u,
                dtype=model_coherence_u.dtype,
                device=model_coherence_u.device,
            )
            data_coherence_v = torch.tensor(
                data_coherence_v,
                dtype=model_coherence_v.dtype,
                device=model_coherence_v.device,
            )
            data_coherence_w = torch.tensor(
                data_coherence_w,
                dtype=model_coherence_w.dtype,
                device=model_coherence_w.device,
            )

        # Compute MSE for each component
        mse_u = torch.mean((model_coherence_u - data_coherence_u) ** 2)
        mse_v = torch.mean((model_coherence_v - data_coherence_v) ** 2)
        mse_w = torch.mean((model_coherence_w - data_coherence_w) ** 2)

        # Average across components
        coherence_loss = (mse_u + mse_v + mse_w) / 3.0

        # Scale by coherence weight parameter
        coherence_loss = self.params.gamma_coherence * coherence_loss

        # Log individual components and total
        self.writer.add_scalar("Coherence Loss/Total", coherence_loss, epoch)
        self.writer.add_scalar("Coherence Loss/U-component", mse_u, epoch)
        self.writer.add_scalar("Coherence Loss/V-component", mse_v, epoch)
        self.writer.add_scalar("Coherence Loss/W-component", mse_w, epoch)

        if coherence_loss.isnan():
            print("Coherence loss is NaN")
            print(f"model_coherence_u: {model_coherence_u}")
            print(f"model_coherence_v: {model_coherence_v}")
            print(f"model_coherence_w: {model_coherence_w}")
            print(f"data_coherence_u: {data_coherence_u}")
            print(f"data_coherence_v: {data_coherence_v}")
            raise ValueError("Coherence loss is NaN")

        return coherence_loss

    def compute_loss_scales(self, epoch, mse_loss, coherence_loss):
        """Compute scaling factors to balance losses."""
        # Initialize loss_scales if it doesn't exist
        if not hasattr(self, "loss_scales") or not self.loss_scales:
            self.loss_scales = {}

        # Initialize on first call or recompute every balance_freq epochs
        if epoch % self.balance_freq == 0 or "mse" not in self.loss_scales:
            # Use exponential moving average of loss magnitudes
            beta = 0.9

            if "mse" not in self.loss_scales:
                # First initialization
                self.loss_scales["mse"] = mse_loss.item()
                self.loss_scales["coherence"] = (
                    coherence_loss.item() if (coherence_loss is not None and coherence_loss != 0) else 1.0
                )
            else:
                # Update existing values
                self.loss_scales["mse"] = beta * self.loss_scales["mse"] + (1 - beta) * mse_loss.item()
                if coherence_loss is not None and coherence_loss != 0:
                    self.loss_scales["coherence"] = (
                        beta * self.loss_scales["coherence"] + (1 - beta) * coherence_loss.item()
                    )

        # Return scaling factors (inverse of magnitude)
        mse_scale = 1.0 / max(self.loss_scales.get("mse", 1.0), 1e-8)
        coh_scale = 1.0 / max(self.loss_scales.get("coherence", 1.0), 1e-8)

        return mse_scale, coh_scale

    def eval(
        self,
        y: torch.Tensor,
        y_data: torch.Tensor,
        theta_NN: torch.Tensor | None,
        epoch: int,
        coherence_data: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Evaluate with component tracking for scheduling."""
        # Compute individual loss components
        mse_loss = self.MSE_term(y, y_data, epoch)

        # Coherence loss
        coherence_loss = torch.tensor(0.0, device=y.device)
        if coherence_data and self.params.gamma_coherence > 0:
            coherence_loss = self.CoherenceLoss(
                coherence_data["model_u"],
                coherence_data["model_v"],
                coherence_data["model_w"],
                coherence_data["data_u"],
                coherence_data["data_v"],
                coherence_data["data_w"],
                epoch,
            )

        # Penalty terms
        pen1_loss = torch.tensor(0.0, device=y.device)
        if self.params.alpha_pen1 > 0:
            pen1_loss = self.Pen1stOrder(y, epoch)

        pen2_loss = torch.tensor(0.0, device=y.device)
        if self.params.alpha_pen2 > 0:
            pen2_loss = self.Pen2ndOrder(y, epoch)

        # Regularization term
        reg_loss = torch.tensor(0.0, device=y.device)
        if self.params.beta_reg > 0 and theta_NN is not None:
            reg_loss = self.Regularization(theta_NN, epoch)

        # Compute total loss
        total_loss = mse_loss + coherence_loss + pen1_loss + pen2_loss + reg_loss

        return total_loss
