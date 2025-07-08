from dotenv import load_dotenv

from ChemCoScientist.frontend import chat, init_page, side_bar
from definitions import CONFIG_PATH

load_dotenv(CONFIG_PATH)

if __name__ == "__main__":
    init_page()
    side_bar()
    chat()
