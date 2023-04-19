"""
Remember to put distinct name of modules, and they should not have same name functions and class inside
Try to use absolute import and reduce cyclic imports to avoid errors
if there are more than one modules then import like this:
from brain_ai import sample_func
"""
from brain_ai import utils as brain_ai_utils


class FrontalLobe:
    def __init__(self, *args, **kwargs):
        self.data_loader = brain_ai_utils.DataLoader(*args, **kwargs)

    def check_types_of_dataset(self):
        pass

    def send_to_specific_type_of_model_handler(self):
        pass

    def compare_models_for_specific_dataset(self):
        pass

    def thinking_the_outcome(self):
        pass

    def tabular_data_handler(self):
        pass

    # def compare_models_results(self):
    #     use prediction-techniques-comparison

    def collecting_results_into_applicable_format(self):
        # get applicable format from prediction-techniques-comparison
        pass

    def save_model(self):
        pass

    def inference(self, data_point):
        pass
