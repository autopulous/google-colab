import roop.globals
import roop.ui as ui


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')

    if not roop.globals.headless:
        ui.update_status(message)
