import subprocess, string, os, datetime, re, sys, itertools

##### Helpers ########
def sanitize(line):
    line_cpy = line.replace(' ', '')
    line_cpy = line_cpy.replace('\n', '')
    return line_cpy
    
def parse_flag(line, keys):
    if line in keys:
        return true
    return false
    
def parse_numeric_param(line):
    return line.split(",")
    
def get_params(params_dict, param_names_list):
    values_lists = []
    for name in param_names_list:
        if name in params_dict.keys():
            values_lists.add(params_dict[name])
        else:
            raise Exception("Config must include parameter: {}".format(name))
    return ret

def parse_dir_param(line):
    return line.replace("\"", '')

def get_files(rel_dir, working_dir):
    full_path = working_dir + rel_dir
    if sys.version[0] == '2':
        out, err = subprocess.Popen(["ls", full_path], stdout=subprocess.PIPE).communicate()
    else:
        out = subprocess.check_output(["ls", full_path]).decode("-utf-8")
    files = out.split("\n")[:-1]
    return files

def normalize_dirs(dirs):
    for key in dirs.keys():
        if dirs[key][-1] != '/':
            dirs[key] += '/'

def create_hpsc_working_dir(dt):
    remote_working_dir = "tmp_" + dt + "/"
    print("Creating " + remote_working_dir + " on hpsc")
    subprocess.Popen(["ssh", "hpscqemu", "mkdir " + remote_working_dir], shell=False).wait()
    return remote_working_dir

def output(s, file1, also_to_console=True):
    if also_to_console:
        print(s)
    file1.write(s + "\n")
    #Not worried about perf, minimal writes
    file1.flush() 

def create_model_if_missing(data_path, num_trees, max_depth, min_leaf_samples):
    fn = str(num_trees) + "_" + str(max_depth) + "_" + str(min_leaf_samples) + ".Model"
    if not data_path[-1] == '/':
        data_path += '/'
    if not os.path.exists(data_path + fn):
        print("Creating missing model file {}".format(data_path + fn))
        subprocess.run(["./create_model",
                        data_path + "train_data.bin",
                        data_path + "train_labels.bin",
                        data_path + "test_data.bin",
                        data_path + "test_labels.bin",
                        num_trees,
                        max_depth,
                        min_leaf_samples],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        subprocess.run(["mv", fn, data_path + fn])
    
_b = sys.version_info[0] < 3 and (lambda x:x) or (lambda x:x.encode('utf-8'))

def ssh_exec_pass(password, args, capture_output=False):
    '''                                                                                  
        Wrapper around openssh that allows you to send a password to                     
        ssh/sftp/scp et al similar to sshpass.                                           
                                                                                         
        Not super robust, but works well enough for most purposes. Typical               
        usage might be::                                                                 
                                                                                         
            ssh_exec_pass('p@ssw0rd', ['ssh', 'root@1.2.3.4', 'echo hi!'])               
                                                                                         
        :param args: A list of args. arg[0] must be the command to run.                  
        :param capture_output: If True, suppresses output to stdout and stores           
                               it in a buffer that is returned                           
        :returns: (retval, output)                                                       
                                                                                         
        *nix only, tested on linux and OSX. Python 2.7 and 3.3+ compatible.              
    '''

    import pty, select
    
    # create pipe for stdout                                                             
    stdout_fd, w1_fd = os.pipe()
    stderr_fd, w2_fd = os.pipe()

    pid, pty_fd = pty.fork()
    if not pid:
        # in child                                                                       
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.dup2(w1_fd, 1)    # replace stdout on child                                   
        os.dup2(w2_fd, 2)    # replace stderr on child                                   
        os.close(w1_fd)
        os.close(w2_fd)

        os.execvp(args[0], args)

    os.close(w1_fd)
    os.close(w2_fd)

    output = bytearray()
    rd_fds = [stdout_fd, stderr_fd, pty_fd]

    def _read(fd):
        if fd not in rd_ready:
            return
        try:
            data = os.read(fd, 1024)
        except (OSError, IOError):
            data = None
        if not data:
            rd_fds.remove(fd) # EOF                                                      

        return data

    # Read data, etc                                                                     
    try:
        while rd_fds:

            rd_ready, _, _ = select.select(rd_fds, [], [], 0.04)

            if rd_ready:

                # Deal with prompts from pty                                             
                data = _read(pty_fd)
                if data is not None:
                    if b'assword:' in data:
                        os.write(pty_fd, _b(password + '\n'))
                    elif b're you sure you want to continue connecting' in data:
                        os.write(pty_fd, b'yes\n')

                # Deal with stdout                                                       
                data = _read(stdout_fd)
                if data is not None:
                    if capture_output:
                        output.extend(data)
                    else:
                        sys.stdout.write(data.decode('utf-8', 'ignore'))

                data = _read(stderr_fd)
                if data is not None:
                    sys.stderr.write(data.decode('utf-8', 'ignore'))
    finally:
        os.close(pty_fd)
        os.close(stderr_fd)
        os.close(stdout_fd)

    pid, retval = os.waitpid(pid, 0)
    return retval, output

algorithms = ['hpsc_dbscan', 'hpsc_rf', 'emu_dbscan', 'emu_rf', 'emusim_dbscan', 'emusim_rf']

def usage():
    print("Usage: python ./test_harness.py config.txt\n"
          "First line of config.txt should name one of : " + str(algorithms))

#From https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def filterForConfig(filepath):
    if len(filepath) < 10:
        return False
    if filepath[-10:] != "config.txt":
        return False
    return True

######################

#If no command line args, run all and exit
if len(sys.argv) == 1:
    test_script = 'test_harness.py'
    config_dir = 'config/'
    print("RUNNING WITH ALL FILES IN CONFIG DIR " + config_dir)
    config_files = getListOfFiles(config_dir)
    for config in config_files:
        if filterForConfig(config):
            print("\n\n++++++++  Running with config file: {}  ++++++++\n\n".format(config))
            subprocess.run(['python3', test_script, config])
        else:
            print("Skipping file: " + config)
    exit(0)

#Otherwise, run from with config file

config = open(sys.argv[1], "r")
config_lines = config.readlines()

#Algorithm to run
ALG = config_lines[0].replace('\n', '') #First line of config file should name the algorithm
if ALG not in algorithms:
    print("Unknown algorithm " + ALG)
    usage()
    exit(1)
    
#For ease of look up
DBSCAN = ALG == 'hpsc_dbscan'   or ALG == 'emu_dbscan' or ALG == 'emusim_dbscan'
RF     = ALG == 'hpsc_rf'       or ALG == 'emu_rf'     or ALG == 'emusim_rf'
HPSC   = ALG == 'hpsc_dbscan'   or ALG == 'hpsc_rf'
EMU    = ALG == 'emu_dbscan'    or ALG == 'emu_rf'
EMUSIM = ALG == 'emusim_dbscan' or ALG == 'emusim_rf'
FIRSTCALL = len(sys.argv) == 2

#Parse input
dir_keys = ['data', 'exec']

numeric_keys = []
#algorithm params
if DBSCAN:
    numeric_keys.extend(['eps', 'minpts', 'bucketsize'])
    if EMUSIM or EMU:
        numeric_keys.extend(['log2_replnodes'])
elif RF:
    numeric_keys.extend(['num_trees', 'tree_depth', 'min_leaf_samples']) 
if HPSC:
    numeric_keys.extend(['procs'])
elif EMUSIM:
    numeric_keys.extend(['log2_num_nodelets', 'log2_memory_size', 'gcs_per_nodelet', 'threads'])
elif EMU:
    numeric_keys.extend(['threads'])
if HPSC and DBSCAN:
    flag_keys = ['simd']
else:
    flag_keys = []
    
working_dir = sys.path[0] + '/'

DEBUG = 0

params = {}
rel_dirs = {}
for line in config_lines[1:]: #First line names algorithm
    line_cpy = sanitize(line)
    pivot = line_cpy.find('=')
    if pivot == -1:
        key = line_cpy
    else:
        key = line_cpy[:pivot]
        data = line_cpy[pivot+1:]
    if key in numeric_keys:
        values = parse_numeric_param(data)
    elif key in dir_keys:
        rel_dir = parse_dir_param(data)
        rel_dirs[key] = rel_dir
        values = get_files(rel_dir, working_dir)
    elif key in flag_keys:
        values = ['0', '1']
    elif key == 'DEBUG':
        data = line_cpy[pivot+1:]
        DEBUG = parse_numeric_param(data)[0]
        continue
    else:
        raise Exception("Unrecognized config line: {}".format(line))
    params[key] = values
        
normalize_dirs(rel_dirs)

if DEBUG:
    print("Running with DEBUG")

#Generate any missing model files for RandomForest
if FIRSTCALL and RF:
    for data_dir in params['data']:
        data_path = rel_dirs['data'] + data_dir
        for num_trees in params['num_trees']:
            for max_depth in params['tree_depth']:
                for min_leaf_samples in params['min_leaf_samples']:
                    create_model_if_missing(data_path, num_trees, max_depth, min_leaf_samples)

now = datetime.datetime.now()
dt = now.strftime("%d-%B-%Y_%H-%M-%S")
date = now.strftime("%d-%B-%Y")
time = now.strftime("%H-%M-%S")

try:
    cred = open("credentials", 'r')
    p = cred.readline()
except:
    p = "" #assume no password

#Create remote working dir
remote_working_dir = "tmp_" + dt + "/"
    
if HPSC:
    if FIRSTCALL:        
        dest = "hpscqemu"
        subprocess.check_output(['ssh', 'hpscqemu', 'echo', '10.0.2.15 qemu1', '>>', '/etc/hosts'])
        subprocess.check_output(['ssh', 'hpscqemu', 'hostname', 'qemu1'])
    else:
        dest = None
elif EMU:    
    if FIRSTCALL:
        dest = 'emu1gw'
    elif len(sys.argv) == 3:
        if sys.argv[2] == 'emu1gw':
            dest = 'emuscb'
        elif sys.argv[2] == 'emuscb':
            dest = 'n0'
        elif sys.argv[2] == 'n0':
            dest = None
elif EMUSIM:
    if FIRSTCALL:
        dest = "SNA-DGX-N01"
    else:
        dest = None
        
if dest is not None:
    print("Creating " + remote_working_dir + " on " + dest)
    ssh_exec_pass(p, ['ssh', dest, "mkdir " + remote_working_dir])
        
#Create local results dir
final_result_dir = working_dir + 'results/' + ALG + '/' + date + '/'
config_base = sys.argv[1][:sys.argv[1].find('.')]
config_base_flattened = config_base.replace('/', '_')
result_fn = config_base_flattened + "_results_" + time + ".txt"
result_path = working_dir + result_fn
if RF:
    accuracy_fn = config_base_flattened + "_accuracy_" + time + ".txt"
elif DBSCAN:
    accuracy_fn = config_base_flattened + "_clusters_" + time + ".txt"
accuracy_path = working_dir + accuracy_fn

#Copy data/executables
remote_dirs = {}
for key in rel_dirs.keys():
    remote_dirs[key] = remote_working_dir + rel_dirs[key]
    files = params[key]
    if dest is not None:
        if sys.version[0] == '3':
            print("Transfering {} files {} from {} to {}:~/{}".format(key, files, rel_dirs[key], dest, remote_dirs[key]))
        ssh_exec_pass(p, ['scp', '-r', working_dir + rel_dirs[key], dest + ":~/" + remote_working_dir])

if dest is not None:
    #Copy to the next destination and rerun...
    this_file_path = sys.argv[0]
    this_file = this_file_path[this_file_path.rfind('/')+1:]
    config_file_path = sys.argv[1]
    config_file = config_file_path[config_file_path.rfind('/')+1:]
    if FIRSTCALL:
        if not os.path.exists(final_result_dir):
            os.makedirs(final_result_dir) #temp path used otherwise
        final_result_path = final_result_dir + result_fn
        final_accuracy_path = final_result_dir + accuracy_fn
        config = config_file_path
    else:
        final_result_path = result_path
        finall_accuracy_path = accuracy_path
        config = config_file
    try:
        print("==== Copying script to " + dest)
        ssh_exec_pass(p, ['scp', this_file_path, dest + ':~/' + remote_working_dir], True)
        print("==== Copying config to " + dest)
        ssh_exec_pass(p, ['scp', config_file_path, dest + ':~/' + remote_working_dir], True)
        print("==== Running script on " + dest)
        ssh_exec_pass(p, ['ssh', dest,
                          'python ' + remote_working_dir + this_file + ' ' + remote_working_dir + config_file + ' ' + dest])
        print("==== Copying results from " + dest)
        ssh_exec_pass(p, ['scp', dest + ':~/' + remote_working_dir + "*results*", final_result_path], True)
        if RF:
            ssh_exec_pass(p, ['scp', dest + ':~/' + remote_working_dir + "*accuracy*", final_accuracy_path], True)
        else:
            ssh_exec_pass(p, ['scp', dest + ':~/' + remote_working_dir + "*clusters*", final_accuracy_path], True)
    finally:
        ssh_exec_pass(p, ['ssh', dest, 'rm', '-r', remote_working_dir])
        exit(0)

add_emusim_path = 'PATH=$PATH:/usr/local/emu/bin; '
    
#Create results file
res_file = open(result_path, "w+")
accuracy_file = open(accuracy_path, "w+")
print("Writing timings to " + result_path)
print("Writing output to " + accuracy_path)

output(ALG, res_file)
output(ALG, accuracy_file, False)

#Build param arrays
param_keys = list(params.keys())
param_vals = []
for key in param_keys:
    param_vals.append(params[key])
last_params = [None] * len(param_keys)    

#Loop over all parameter combinations
for current_params in itertools.product(*param_vals):
    get = lambda key : str(current_params[param_keys.index(key)])

    #print any params that changed this iteration
    for i in range(len(current_params)):
        if current_params[i] is not last_params[i]:
            output(i * '\t' + '{} = {}'.format(param_keys[i], current_params[i]), res_file)
            output(i * '\t' + '{} = {}'.format(param_keys[i], current_params[i]), accuracy_file, False)

    #Build the command...

    #Environment
    if HPSC:
        command = "aarch64-poky-linux-mpirun"
        command += " -np " + get('procs')
        command += " -hosts 10.0.2.15 "
    elif EMU:
        command = "emu_multinode_exec 10000 -- "
    elif EMUSIM:
        command = "emusim.x"
        command += " --log2_num_nodelets " + get('log2_num_nodelets')
        command += " -m " + get('log2_memory_size')
        command += " --gcs_per_nodelet " + get('gcs_per_nodelet')
        command += " -- "

    #Executable
    command += working_dir + rel_dirs['exec'] + get('exec')

    #Data
    data_prefix = working_dir + rel_dirs['data'] + get('data')

    if DBSCAN:
        command += " -i " + data_prefix
    elif RF:
        model_file = get('num_trees') + '_' + get('tree_depth') + '_' + get('min_leaf_samples') + ".Model"
        fns = ['train_data.bin', 'train_labels.bin', 'test_data.bin', 'test_labels.bin', model_file]
        for filename in fns:
            command += " " + data_prefix + '/' + filename

    #Remaining algorithm parameters
    if DBSCAN:            
        command += " -m " + get('minpts')
        command += " -e " + get('eps')
        if HPSC:
            command += " -c " + get('bucketsize')
            if 'simd' in param_keys:
                command += " -s " + get('simd')
        elif EMU or EMUSIM:
            command += " -s " + get('bucketsize')
            command += " -t " + get('threads')
            command += " -r " + str(2**int(get('log2_replnodes')))
            command += " -b "
    elif RF:
        command += " " + get('num_trees')
        command += " " + get('tree_depth')
        if HPSC:
            command += " " + get('min_leaf_samples')
        elif EMUSIM or EMU:
            command += " 0" #Don't limit number of samples
            if int(get('num_trees')) % int(get('threads')) != 0:
                print("Number of threads must evenly divide number of trees!")
                output('N/A', res_file)
                output('N/A', accuracy_file, False)
                last_params = current_params
                continue
            treesPerThread = int(max(int(get('num_trees')) / int(get('threads')), 1)) #TODO
            
            command += " " + str(treesPerThread)
                
    #Run the command
    print("\t" * len(current_params) + "Executing remote command: " + command)
    out = ""

    if EMUSIM:
        command = add_emusim_path + command

    try:
        out = subprocess.check_output([command], shell=True).decode("utf-8")
        if HPSC:
            if DBSCAN:
                runtime = re.search("(?<=Parallel DBSCAN \(init, local computation, and merging\) took )\d+\.\d+", out).group(0)
                clusters = re.search("(?<=Total number of clusters )\d+", out).group(0)
            elif RF:
                num_samples = re.search("(?<=Test samples: )\d+", out).group(0)
                total_runtime = re.search("(?<=Predict took: )\d+\.\d+", out).group(0)
                error_rate = float(re.search("(?<=the total error rate is:)\d+\.\d+", out).group(0))
                runtime = str(float(total_runtime) / float(num_samples))
        elif EMU or EMUSIM:
            if DBSCAN:
                runtime = re.search("(?<=DBSCAN .total. took )\d+", out.decode("utf-8")).group(0)
#                clusters = re.search("(?<=Number of clusters: )\d+", out).group(0)
                clusters = 0
            elif RF:
                num_samples = re.search("(?<=Testing samples: )\d+", out).group(0)
                total_runtime = re.search("(?<=Testing took )\d+", out.decode("utf-8")).group(0)
                error_rate = float(re.search("(?<=error rate )\d+\.\d+", out).group(0))
                runtime = str(float(total_runtime))
                runtime = str(float(total_runtime) / float(num_samples))
        if RF:
            value = 1 - error_rate
        else:
            value = clusters
    except Exception as e:
        print(e)
        runtime = 'N/A'
        value = 'N/A'
        
    if DEBUG:
        print(out)
        
    output("\t" * len(current_params) + runtime, res_file)
    output("\t" * len(current_params) + str(value), accuracy_file)
    #prepare for next loop iter  
    last_params = current_params
