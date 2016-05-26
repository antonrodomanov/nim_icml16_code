import os
from time import time, sleep

datasets = [
    {'name': 'a9a', 'n': 32561, 'd': 123},
    {'name': 'mushrooms', 'n': 8124, 'd': 112},
    {'name': 'ijcnn1', 'n': 49990, 'd': 22},
    {'name': 'cod-rna', 'n': 59535, 'd': 8},
    {'name': 'covtype', 'n': 581012, 'd': 54},
    {'name': 'w8a', 'n': 49749, 'd': 300},
    {'name': 'protein', 'n': 145751, 'd': 74},
    {'name': 'quantum', 'n': 50000, 'd': 65},
    {'name': 'SUSY', 'n': 5000000, 'd': 18},
    {'name': 'alpha', 'n': 500000, 'd': 500},
]
reg_type = 'l1' # 'l1' or 'l2'
waiting_time = 2

add_params = '' if reg_type == 'l2' else '--lambda 0'
nim_minibatch_size = 100 if reg_type == 'l2' else 5000

#####################################################################
# Run all methods
#####################################################################
for dataset in datasets:
    start_t = time()
    os.system('./main --dataset %s --method NIM %s --minibatch_size %d --max_epochs 100' % (dataset['name'], add_params, nim_minibatch_size))
    elaps_t = time() - start_t

    max_time = 3 * elaps_t

    os.system('./main --dataset %s --method SAG %s --minibatch_size 10 --max_epochs 10000 --opt_allowed_time %g' % (dataset['name'], add_params, max_time))
    os.system('./main --dataset %s --method newton %s --exact 0 --max_epochs 500 --opt_allowed_time %g' % (dataset['name'], add_params, max_time))
    if reg_type == 'l2':
        os.system('./main --dataset %s --method LBFGS --max_epochs 500 --opt_allowed_time %g' % (dataset['name'], max_time))

    print('\n\nWait for %d seconds...\n\n' % waiting_time)
    sleep(waiting_time)


#####################################################################
# Diferent minibatch size: NIM and SAG
#####################################################################
# minibatch_sizes = [2, 10, 100, 1000, 5000, 10000, 30000]

# for dataset in datasets:
#     for minibatch_size in minibatch_sizes:
#         if minibatch_size > dataset['n']: continue

#         os.system('./main --dataset %s --method NIM %s --minibatch_size %d --max_epochs 100' % (dataset['name'], add_params, minibatch_size))
#         #os.system('./main --dataset %s --method SAG %s --minibatch_size %d --max_epochs 500' % (dataset['name'], add_params, minibatch_size))

#         print('\n\nWait for %d seconds...\n\n' % waiting_time)
#         sleep(waiting_time)