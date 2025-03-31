import os
import json
import sys
import numpy as np
import projection_BNS.utils as utils

def main():
    
    first_dir = os.path.abspath(sys.argv[1])
    second_dir = os.path.abspath(sys.argv[2])
    
    # # Get the templates of second_dir (they are one directory up)
    # second_dir_up = os.path.dirname(os.path.dirname(second_dir))
    second_dir_up = os.path.dirname(second_dir)
    
    # Copy template.ini and template.prior to the second_dir, but name them config.ini and prior.prior
    template_ini = os.path.join(second_dir_up, "template.ini")
    template_prior = os.path.join(second_dir_up, "template.prior")
    template_generation_prior = os.path.join(second_dir_up, "template_generation.prior")
    
    new_ini = os.path.join(second_dir, "config.ini")
    new_prior = os.path.join(second_dir, "prior.prior")
    new_template_generation = os.path.join(second_dir, "generation_prior.prior")
    
    if not os.path.exists(template_ini):    
        raise ValueError("template.ini not found in directory {}".format(second_dir_up))
    if not os.path.exists(template_prior):
        raise ValueError("template.prior not found in directory {}".format(second_dir_up))
    
    os.system(f"cp {template_ini} {new_ini}")
    os.system(f"cp {template_prior} {new_prior}")
    os.system(f"cp {template_generation_prior} {new_template_generation}")
    
    new_eos_filename = utils.get_eos_file_from_dirname(second_dir)
    new_m, _, new_l = utils.load_eos(new_eos_filename)
    
    if not os.path.exists(first_dir):
        raise ValueError("First directory not found")
    
    if not os.path.exists(second_dir):
        print(f"Second directory {second_dir} not found -- creating it now")
        os.makedirs(second_dir)
    
    injection_filename = os.path.join(first_dir, "injection.json")
    
    # Locate the file
    if not os.path.exists(injection_filename): 
        raise ValueError("Injection file not found in directory {}".format(first_dir))
    
    # Load the file
    with open(injection_filename, "r") as f:
        injection = json.load(f)
        
    # Adjust lambda1 and lambda2 based on the new EOS based on interpolation
    m1 = injection["mass_1_source"]
    m2 = injection["mass_2_source"]
    
    lambda_1 = np.interp(m1, new_m, new_l)
    lambda_2 = np.interp(m2, new_m, new_l)
    
    injection["lambda_1"] = lambda_1
    injection["lambda_2"] = lambda_2
    
    # Save the new injection file
    new_injection_filename = os.path.join(second_dir, "injection.json")
    with open(new_injection_filename, "w") as f:
        json.dump(injection, f)
        

if __name__ == "__main__":
    main()