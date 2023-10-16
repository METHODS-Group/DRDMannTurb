"""
This module sets up the logging utilities to be used throughout
"""

import logging as log

drdmannturb_log = log.getLogger()

drdmannturb_log.setLevel(log.INFO)
log.addLevelName(5, "OPTINFO")
setattr(log, "OPTINFO", 5)



def equals_border_fprint(msg: str, loc: str = "") -> None:
    """
    Format print routine with = vertical delimiting. Specifically, prints out
    in the form

    ```
    =================================
    [{loc}] -> {msg}
    =================================
    ```

    where the substring `[{loc}] -> ` is added only if loc is not empty.

    Parameters
    ----------
    msg
        The intended message
    loc
        A string to be printed between square brackets preceding the
        arrow. Intended to be used as some indication of where the 
        call to this function is located. By default, the empty string.
    """

    header = ""
    if loc is not "":
        header = f"[{loc}] -> "

    print(f"\n=================================\n"
        f"{header}{msg}\n"
        "=================================\n"
    )


def simple_fprint(msg: str, loc: str = "", tabbed: bool = False) -> None:
    """
    Format print routine. Specifically, prints out in the form

    ```
    {tab}[{loc}] -> msg
    ```

    Parameters
    ----------
    msg
        The intended message.
    loc
        A string to be printed between square brackets preceding the
        arrows. Intended to be used as some indication of where the
        call to this function is located. By default, the empty string.
    tabbed
        If true, prepends a tab's worth of white space (4 spaces).
        By default, False.
    """

    header = "    " if tabbed else ""
    if loc is not "":
        header += f"[{loc}] -> "

    print(f"{header}{msg}")


setattr(log.getLoggerClass(), "optinfo", equals_border_fprint)
setattr(log.getLoggerClass(), "simple_optinfo", simple_fprint)
