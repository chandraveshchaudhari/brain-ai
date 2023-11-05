import logging
import os
from datetime import datetime


class Logger(logging.Logger):

    def __init__(self, log_project_name: str = 'Project Name', log_directory_path=None):
        super().__init__(log_project_name)
        self.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if log_directory_path is not None:
            log_filename = os.path.join(log_directory_path, f'{log_project_name}_{current_time}.log')
        else:
            log_filename = f'{log_project_name}_{current_time}.log'
        self.file_handler = logging.FileHandler(log_filename)
        self.file_handler.setFormatter(self.formatter)
        self.addHandler(self.file_handler)
        self.welcome_log()

    def welcome_log(self, name: str = 'BrainAutoML', full_length=80):
        title_length = len(name)
        full_length += title_length
        half_padding = (full_length - title_length) // 2
        self.info(f"\n"
                  f"{'*'* full_length}\n"
                  f"{'*'* full_length}\n"
                  f"**{' '* (full_length-4)}**\n"
                  f"**{' '* (full_length-4)}**\n"
                  f"**{' ' * (half_padding -2)}{name}{' ' * (half_padding -2)}**\n"
                  f"**{' '* (full_length-4)}**\n"
                  f"**{' '* (full_length-4)}**\n"
                  f"{'*'* full_length}\n"
                  f"{'*'* full_length}\n"
                  f"\n")
