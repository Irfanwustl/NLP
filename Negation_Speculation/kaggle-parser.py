# Usage: Ctrl + A on a kaggle version, paste it in kaggle-log.txt.
# This code appends results to output.csv

import re
import csv

out_data = []
model = "base"
subtask = None
should_capture_scores = False
test_set = None
run = 0
macrotask = "negation"
num_test_combos_done = 0

# For scope resolution
NUM_TEST_SETS = 2

with open("kaggle-log.txt") as file:
    for line in file:
        res = re.match("CUE_MODEL = ([^ ]*) ", line)
        if res:
            group = res.group(1)
            match group:
                case "PRETRAINED_PATH_augnb":
                    model = "AugNB"
                case "PRETRAINED_PATH_cuenb":
                    model = "CueNB"
                case _:
                    raise RuntimeError(f"unknown group {group}")
        # if re.match("CUE_MODEL =", line):
        #     if "aug" in line:
        #         model = "AugNB"
        #     if "cue" in line:
        #         model = "CueNB"
        res = re.match("SUBTASK = '(.*)'", line)
        if res:
            group = res.group(1)
            match group:
                case "cue_detection":
                    subtask = "cue"
                case "scope_resolution":
                    subtask = "scope"
                case _:
                    raise RuntimeError(f"unknown group {group}")
        if line.strip() == "Early stopping":
            # Training is done
            should_capture_scores = True
        res = re.match("Evaluate on (.*):", line)
        if res:
            group = res.group(1)
            match group:
                case "bioscope_abstracts":
                    test_set = "BA"
                case "bioscope_full_papers":
                    test_set = "BF"
                case "bioscope_full_papers_punct" | "bioscope_full_papers_no_punct" | "bioscope_abstracts_punct" | "bioscope_abstracts_no_punct":
                    # Ignore these, they are used to debug the model
                    test_set = None
                case _:
                    raise RuntimeError(f"unknown group {group}")
        if should_capture_scores and test_set and re.match("F1 Score: ", line):
            score = line.split()[-1]
            row = [model, macrotask, subtask, test_set, run, score]
            # print(row)
            out_data.append(row)
            # Update macro task to be the other one
            # For scope resolution, only do this after all its test sets are evaluated
            # since all the negation ones are evaluated first, then all the speculation ones
            num_test_combos_done += 1
            if subtask == "cue" or num_test_combos_done == NUM_TEST_SETS:
                macrotask = "negation" if macrotask == "speculation" else "speculation"
                num_test_combos_done = 0
        if "DONE! **" in line:
            run += 1
            # Don't capture logs before training is done
            should_capture_scores = False
            test_set = None
            num_test_combos_done = 0

print(model)
print(subtask)

with open("output.tsv", "a") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(out_data)
