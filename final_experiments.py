from ClassifierGenerationGuided import run_main_CTRL_guided
from ClassifierGenerationGuidedBeam import run_main_CTRL_beam

run_main_CTRL_beam(data_augmentation = False, do_unlikelihood = False, classifier_path = "classifier_50t_aug", output_path = "CTRL_beam_likelihood_1", inp_epochs = 10, lr=5e-6)
run_main_CTRL_guided(data_augmentation = False, do_unlikelihood = False, classifier_path = "classifier_50t_aug", output_path = "CTRL_likelihood_1", inp_epochs = 10, lr=5e-6)
run_main_CTRL_beam(data_augmentation = False, do_unlikelihood = True, classifier_path = "classifier_50t_aug", output_path = "CTRL_beam_unlikelihood_1", inp_epochs = 10, lr=5e-6)
run_main_CTRL_guided(data_augmentation = False, do_unlikelihood = True, classifier_path = "classifier_50t_aug", output_path = "CTRL_unlikelihood_1", inp_epochs = 10, lr=5e-6)
run_main_CTRL_beam(data_augmentation = False, do_unlikelihood = False, classifier_path = "classifier_50t_aug", output_path = "CTRL_beam_likelihood_2", inp_epochs = 10, lr=5e-5)
run_main_CTRL_guided(data_augmentation = False, do_unlikelihood = False, classifier_path = "classifier_50t_aug", output_path = "CTRL_likelihood_2", inp_epochs = 10, lr=5e-5)
run_main_CTRL_beam(data_augmentation = False, do_unlikelihood = True, classifier_path = "classifier_50t_aug", output_path = "CTRL_beam_unlikelihood_2", inp_epochs = 10, lr=5e-5)
run_main_CTRL_guided(data_augmentation = False, do_unlikelihood = True, classifier_path = "classifier_50t_aug", output_path = "CTRL_unlikelihood_2", inp_epochs = 10, lr=5e-5)