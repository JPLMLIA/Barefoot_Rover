import matplotlib.pyplot as plotlib
import matplotlib.patches as mpatches
import re, sys, os, datetime, math, subprocess, copy

#From https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories
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

usage = "Usage: \"python plot.py results.txt\"\n" + "Or\n" + "\"python plot.py <results1.txt config1.txt> <results2.txt config2.txt>...\""

if len(sys.argv) > 2 and len(sys.argv) % 2 == 0:
    print(usage)
    exit(1)

gfigs = {}
gaxess = {}
should_plot = {}
    
#gaxes.set_ylabel("cycles") #TODO: get from data (cycles/seconds)
parameter_to_plot = None
plot_global = True

if len(sys.argv) == 2:
    measured = None
    if sys.argv[1].endswith(".txt"):
        if os.path.isdir(sys.argv[1][:-4]):
            exit(0) #Already run
        elif "clusters" in sys.argv[1]:
            measured = "clusters"        
        elif "accuracy" in sys.argv[1]:
            measured = "accuracy"
        elif "results" in sys.argv[1]:
            measured = "runtime"
    if measured is not None:
        print("Running on result file {}".format(sys.argv[1]))
    else:
        #Run on all files in dir arg
        if os.path.isdir(sys.argv[1]) and not os.path.exists("./" + sys.argv[1] + ".txt"):
            print("+++++++++ Running on all results in dir {} ++++++++".format(sys.argv[1]))
            results_files = getListOfFiles(sys.argv[1])
            for result in results_files:
                out = subprocess.Popen(['python', sys.argv[0], result])
        else:
            print("Skipping {}".format(sys.argv[1]))
        exit(0)

dt = datetime.datetime.now()

resultpath = sys.argv[1]
outdir = resultpath[:resultpath.find('.txt')] + '/'

os.mkdir(outdir)
readme = open(outdir + "info.txt", "w+")

#Get parameter variation across all files
#TODO: Refactor this block...
####################################################################
all_param_values = {}
for i in range(math.floor(len(sys.argv)/2)):
    params = []
    locked_params = []
    results_filename = sys.argv[2*i+1]
    results_file = open(results_filename, 'r')
    results_lines = results_file.readlines()
    results_file.close()
    use_config = len(sys.argv) > 2
    if use_config:
        config_file = open(sys.argv[2*i+2], 'r')
        config_lines = config_file.readlines()
        config_file.close()
    else:
        config_lines = [] #No config
    #get the parameters that varied for this run
    for line in results_lines:
        line_copy = line.replace('\t', '')
        line_copy = line_copy.replace(' ', '')
        if line_copy[0].isalpha():
            index = line_copy.find('=')
            p = line_copy[:index]
            params.append(p)
            if p not in all_param_values.keys():
                all_param_values[p] = []
        else:
            break

#parse the rest of config file
    for line in config_lines:
        line_cpy = line.replace(' ', '')
        if line_cpy[0] is 'x' or line_cpy[0] is 'X':
            continue
        else:
            index = line_cpy.find('=')
            param_var = line_cpy[:index]
            if param_var not in all_param_values.keys():
                all_param_values[param_var] = []
            for value in line_cpy[index+1:-1].split(','):
                if value not in all_param_values[param_var]:
                    all_param_values[param_var].append(value)
            locked_params.append(param_var)

#track the current parameter config and process input text file
    for line in results_lines:
        l = line.replace('\t','')
        for p in params:
            if p in l:
            #update current param
                x = l[l.find(p)+len(p)+3:-1]
                if x not in all_param_values[p] and p not in locked_params:
                    all_param_values[p].append(x)

def setYLabel(axes, algorithm):
    if "runtime" in measured and algorithm is not None:
        if 'emusim' in algorithm:
            label = "cycles"
            if 'rf' in algorithm:
                label += " per sample"
        elif 'emu' in algorithm or 'hpsc' in algorithm:
            label = "time (s)"
    else:
        label = measured
    axes.set_ylabel(label)

####################################################################
algos = ['emu_dbscan', 'emu_rf', 'emusim_dbscan', 'emusim_rf', 'hpsc_dbscan', 'hpsc_rf']

for i in range(math.floor(len(sys.argv)/2)):

    file = None
    param_values = {}
    params = []
    locked_params = []

    results_filename = sys.argv[2*i+1]
    results_file = open(results_filename, 'r')
    results_lines = results_file.readlines()
    algorithm = results_lines[0].replace('\n', '')
    if algorithm in algos:
        results_lines = results_lines[1:]
    else:
        algorithm = None
        
    results_file.close()
    if use_config:
        config_file = open(sys.argv[2*i+2], 'r')
        config_lines = config_file.readlines()
        config_file.close()
    else:
        config_lines = [] #No config

#make copy of raw data with plots
    subprocess.run(["cp", results_filename, outdir + results_filename.replace('/', '___')])

    readme.write("Result {}\n".format(results_filename))
    if use_config:
        readme.write("\tConfig:\n")
        for line in config_lines:
            readme.write("\t\t{}".format(line))
    else:
        readme.write("\tConfig:  AUTO\n")

#get the parameters that varied for this run
    for line in results_lines:
        line_copy = line.replace('\t', '')
        line_copy = line_copy.replace(' ', '')
        if line_copy[0].isalpha() and not 'N/A' in line_copy:
            index = line_copy.find('=')
            params.append(line_copy[:index])
        else:
            break

    print(params)

#init param_values 2d array
    for i in range(len(params)):
        param_values[params[i]] = []

#parse the rest of config file
    for line in config_lines:
        line_cpy = line.replace(' ', '')
        line_cpy = line_cpy.replace('\n', '')
        if line_cpy[:7] == 'results':
            continue
        elif line_cpy[0] is 'x' or line_cpy[0] is 'X':
            line_cpy = line_cpy.replace('=', '')
            if parameter_to_plot != None and parameter_to_plot != line_cpy[1:]:
                plot_global = False
            parameter_to_plot = line_cpy[1:]
            gaxes.set_xlabel(parameter_to_plot)
        else:
            index = line_cpy.find('=')
            param_var = line_cpy[:index]
            if param_var in params:
                param_values[param_var] = line_cpy[index+1:].split(',')
                locked_params.append(param_var)
            else:
                print("No parameter: %s" % param_var)
                exit(1)

#the timing values
    timings = {}
    already_plotted = []

#track the current parameter config and process input text file
    current_params = [None] * len(params)
    for line in results_lines:
        l = line.replace('\t','')
        isDataLine = True
        for p in params:
            if p in l:
            #update current param
                x = l[l.find(p)+len(p)+3:-1]
                current_params[params.index(p)] = x
                if x not in param_values[p] and p not in locked_params:
                    param_values[p].append(x)
                isDataLine = False
        if isDataLine:
        #get timing value
            timings[tuple(current_params)] = l

    readme.write("\tParameters:\n")
    for key in param_values.keys():
        readme.write("\t\t{}: {}\n".format(key, param_values[key]))

    def getCaption(values):
        line_len = 0
        caption = ""
        for i in range(len(params)):
            if values[i] is None:
                continue
            line_len += len(params[i] + '=' + values[i])
            if line_len > 30:
                line_len = 0
                caption += '\n'
            caption += params[i] + '=' + values[i] + "   "
        return caption[:-3]

    for line in results_lines:
        l = line.replace('\t','')
        isDataLine = True
    #Get Parameters
        for p in params:
            if p in l:
                x = l[l.find(p)+len(p)+3:-1]
                current_params[params.index(p)] = x
                isDataLine = False

    #Plot (potential) new set of parameters
        if isDataLine:
            if use_config:
                if parameter_to_plot is not None:
                    params_to_plot = [parameter_to_plot]
                else:
                    params_to_plot = [p for p in params if p not in locked_params]
            else:
                params_to_plot = params
            for param_to_plot in params_to_plot:
                #check for valid params
                if len(param_values[param_to_plot]) < 4: #minimum 4 data points to plot
                    continue
                skip = False
                fn = ""
                plot_key = ""
                for i in range(len(params)):
                    if current_params[i] not in param_values[params[i]]:
                        skip = True
                        break
                    if params[i] == param_to_plot:
                        continue
                    elif len(all_param_values[params[i]]) > 1:
                        fn = fn + params[i] + "=" + current_params[i] + "__"
                        plot_key += params[i] + "=" + current_params[i] + " "
                if skip or plot_key in already_plotted:
                    print("Skipping " + str(current_params))
                    continue
                else:
                    print("Plotting " + str(current_params))
                already_plotted.append(plot_key)

                #Get data and plot
                xData = param_values[param_to_plot]
                yData = []
                xDataNoMissing = []
                for x in xData:
                    current_params[params.index(param_to_plot)] = x
                    try:
                        yData.append(float(timings[tuple(current_params)]))
                        xDataNoMissing.append(x)
                    except:
                        continue
                try:
                    x = list(map(float, xDataNoMissing))
                except:
                    print("Skipping " + str(current_params) + "\n\t" + str(xDataNoMissing))
                    continue
                if len(x) is 0:
                    print("Skipping " + str(current_params) + "\n\t" + str(xDataNoMissing))
                    continue
                fig, axes = plotlib.subplots()
                axes.set_yscale('linear')
                y = yData
                axes.plot(x, y, 'o', x, y, 'k')

                #Configure plot
                values = copy.deepcopy(current_params)
                values[params.index(param_to_plot)] = None
                axes.set_xlabel(param_to_plot + "\n\n" + getCaption(values))
                setYLabel(axes, algorithm)
                #plotlib.legend()

                #Plot!
                subdir = outdir + param_to_plot + '_X/'
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                plotlib.title(algorithm.replace('_', ' ') + " " + measured + " vs " + str(param_to_plot))
                plotlib.tight_layout()
                fig.savefig(subdir + fn[:-2] + '.png')
                plotlib.close(fig)
                if plot_global:
                    #Loop over all secondary parameters
                    for i in range(len(current_params)):
                        if i == params.index(param_to_plot):
                            continue
                        key = current_params[:] #copy
                        label = params[i] + " = " + key[i]
                        key[params.index(param_to_plot)] = 'X'
                        key[i] = 'S' #secondary
                        key = tuple(key)

                        if key not in gfigs:
                            gfigs[key] = []
                            gaxess[key] = param_to_plot
                        
                        gfigs[key].append((x, y, label)) #do plots one figure at a time later to avoid >20 plots open
    
    def getLabel(figureTuple):
        lbl = figureTuple[2]
        i = lbl.find('=') + 2
        if lbl[i].isnumeric():
            return float(lbl[i:])
        else:
            return lbl

    for key in gfigs.keys():
        if len(gfigs[key]) > 1:
            gfigs[key].sort(key=getLabel) #sort by label
            global_fig, global_axes = plotlib.subplots()
            for (x, y, lbl) in gfigs[key]:
                global_axes.plot(x, y, '-k')
                global_axes.plot(x, y, '-o', label=lbl)
            fn = ''
            values = []
            for i in range(len(params)):
                if key[i] is 'X':
                    subdir = outdir + params[i] + '_X/'
                    #TODO: dont overlap legend
                    #plotlib.title(algorithm.replace('_', ' ') + " runtime vs " + str(params[i]))
                    values.append(None)
                elif key[i] is 'S':
                    subsubdir = params[i] + '_grouped/'
                    values.append(None)
                elif len(all_param_values[params[i]]) > 1:
                    fn = fn + params[i] + "=" + key[i] + "__"
                    values.append(key[i])
                else:
                    values.append(all_param_values[params[i]][0])
            if not os.path.exists(subdir + subsubdir):
                os.makedirs(subdir + subsubdir)

            global_axes.set_xlabel(gaxess[key] + '\n\n' + getCaption(values))
            setYLabel(global_axes, algorithm)
            global_fig.legend()

            plotlib.tight_layout()

            global_fig.savefig(subdir + subsubdir + fn[:-2] + '.png')
            plotlib.close(global_fig)
