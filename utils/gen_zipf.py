import os, sys
import numpy as np
import pandas as pd
import random
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: %s <save_path> <num_resources> <num_samples> <zipf_param> <num_progs>" % sys.argv[0])
        exit(0)
    save_path = sys.argv[1]
    #total number of files that need to be created,resources
    num_files = int(sys.argv[2])
    #total number of requests that needs to be simulated
    num_samples = int(sys.argv[3])
    #How uneven the popularity is. Higher = more unfair ie some files will be having high popularity
    param = float(sys.argv[4])
    # no of users making requests
    num_progs = int(sys.argv[5])

    df = None
    for i in range(num_progs):
        files = np.arange(num_files)
        # Random ranks. Note that it starts from 1.
        ranks = np.random.permutation(files) + 1

        # Distribution,Assign Popularity Using Zipf's Law
        # files with lower ranks ,gets higher popularity and bigger ranks (rank 100) get lower probability.
        pdf = 1 / np.power(ranks, param)
        pdf /= np.sum(pdf)

        # Draw samples
        #Pick files Based on Popularity
        requests = np.random.choice(files, size=num_samples, p=pdf)
        operations = np.full_like(requests, 0)
        executions = np.full_like(requests, 1)

        # Make dataframe and save .csv
        tmp = pd.DataFrame({'blocksector': requests, 'read/write': operations, 'boot/exec': executions})
        if df is None:
            df = tmp
        else:
            df = pd.concat((df, tmp), axis=0)

    # Save
    df.to_csv(save_path, index=False, header=True)
