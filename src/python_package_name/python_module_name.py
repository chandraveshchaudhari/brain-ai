"""
Remember to put distinct name of modules and they should not have same name functions and class inside
Try to use absolute import and reduce cyclic imports to avoid errors
if there are more than one modules then import like this:
from python_package_name import sample_func
"""
from python_package_name import second_python_module_name

run_this = second_python_module_name.sample_func("well anything")

if __name__ == '__main__':
    print(run_this)

