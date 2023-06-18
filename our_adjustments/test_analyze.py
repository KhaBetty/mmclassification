import json
# Load the data from json files
basic_path = '/home/maya/projA/runs'
orig_folder= '/resized_64'
file_name = '/test_results.json'
orig_path = basic_path + orig_folder +file_name
mean_diff = {}
mean_all = {}
with open(orig_path) as f:
	data = json.load(f)
	data = data['pred_score']
	#calculate the mean of the data
	mean = sum(data)/len(data)
	mean_all[orig_folder] = mean
	folders = ['/cropped_shuffle_64_fixed_AVG_sep_6','/cropped_shuffle_64_fixed_max_sep_6']
	#defferent from list
	for folder in folders:
         with open(basic_path + folder +file_name) as tmp:
            data_tmp = json.load(tmp)
            data_tmp = data_tmp['pred_score']
            mean_tmp = sum(data_tmp)/len(data_tmp)
            mean_all[folder] = mean_tmp
            mean_diff[folder] =  mean_tmp - mean
        #save the data in new json file
with open(basic_path +  '/mean_diff.json', 'w') as m:
    json.dump(mean_diff, m, indent=4)
with open(basic_path +  '/mean_all.json', 'w') as m:
    json.dump(mean_all, m, indent=4)
