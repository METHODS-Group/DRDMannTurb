import colorama
import matplotlib.pyplot as plt
from colorama import Fore, Style
from seaborn import heatmap

colorama.init(autoreset=True)


def print_header(text):
    """Print a formatted header with color"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(80)}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 80}{Style.RESET_ALL}")


def print_section(text):
    """Print a formatted section header with color"""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}{'-' * 40}")
    print(f"{Fore.GREEN}{Style.BRIGHT}{text}")
    print(f"{Fore.GREEN}{Style.BRIGHT}{'-' * 40}{Style.RESET_ALL}")


def print_param(name, value, unit=""):
    """Print a parameter with color coding"""
    unit_str = f" {unit}" if unit else ""

    if isinstance(value, (int, float)):
        # For integers, use comma as thousand separator
        if isinstance(value, int):
            formatted_value = f"{value:,}"
        # For floats with decimal places
        elif value.is_integer():
            formatted_value = f"{int(value):,}"
        else:
            # Extract the format precision if it exists in the value string
            if isinstance(value, str) and "." in value:
                parts = value.split(".")
                if len(parts) == 2 and parts[1].isdigit():
                    precision = len(parts[1])
                    formatted_value = f"{float(value):,.{precision}f}"
                else:
                    formatted_value = value
            else:
                # Default formatting for floats
                formatted_value = f"{value:,.2f}" if isinstance(value, float) else value
    else:
        # For non-numeric values, use as is
        formatted_value = value

    print(f"{Fore.YELLOW}{name}: {Fore.WHITE}{Style.BRIGHT}{formatted_value}{unit_str}")


def arr_debug(arr, name, plot_heatmap=False):
    """Enhanced array debug function with color coding"""
    print(f"\n{Fore.MAGENTA}Debug info for: {Style.BRIGHT}{name}{Style.RESET_ALL}")
    print(f"  {Fore.BLUE}Shape: {Fore.WHITE}{arr.shape}")
    print(f"  {Fore.BLUE}Range: {Fore.WHITE}{arr.min():.6f} to {arr.max():.6f}")
    print(f"  {Fore.BLUE}Mean: {Fore.WHITE}{arr.mean():.6f}, {Fore.BLUE}Std: {Fore.WHITE}{arr.std():.6f}")

    if plot_heatmap:
        plt.figure(figsize=(10, 8))
        plt.title(f"Heatmap of {name}")
        heatmap(arr)
        plt.show()
