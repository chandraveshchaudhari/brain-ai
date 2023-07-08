from brain_ai.memory import Memory


class Brain:
    def __init__(self, memory_directory_path=None, *args, **kwargs):
        self.memory = Memory(memory_directory_path)
        self.args = args
        self.kwargs = kwargs

    def generate_brain_configuration_file(self, output_configuration_file_path=None):
        config_file_path = self.memory.configuration_path if output_configuration_file_path is None else output_configuration_file_path
        self.memory.generate_configuration_file(config_file_path)

    def train(self):
        for dataset in self.memory.configuration['datasets']:
            for data_type, data in dataset.items():
                if data_type == 'Tabular_data':
                    self.train_tabular_data(data)
                elif data_type == 'Sentiment_data':
                    self.train_sentiment_data(data)
                elif data_type == 'Image_data':
                    self.train_image_data(data)
                elif data_type == 'Text_data':
                    self.train_text_data(data)
                elif data_type == 'Audio_data':
                    self.train_audio_data(data)
                elif data_type == 'Video_data':
                    self.train_video_data(data)
                elif data_type == 'Time_series_data':
                    self.train_time_series_data(data)
                else:
                    raise ValueError(f"Invalid data type {data_type}")








