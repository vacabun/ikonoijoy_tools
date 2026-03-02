import yaml
from bot import ADBGameBot


def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    bot = ADBGameBot(config)
    bot.create_gui()
