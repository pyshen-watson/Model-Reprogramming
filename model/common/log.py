from colorama import Fore, Style

def log_success(message: str):
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def log_fail(message: str):
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")
    raise ValueError()
