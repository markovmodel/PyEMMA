import os
import numpy
import sys
from time import gmtime, strftime
import emma_settings as settings
import emma_msm_mockup as msm

class ModelBuilder:
    
    def __init__(self,
                 transformers,
                 #clustering,
                 #discretizer,
                 proteintraj,
                 lag = 1,
                 reversible = True,
                 connected_set = True,
                 calculate_stationary = True,
                 calculate_kinetics = True,
                 dir_simulation_data = "./results/",
                 dir_model = "./model/",
                 out = sys.stdout,
                 ):
        """
        """
        
        self.transformers = transformers
        #self.transformer = transformer
        #self.clustering = clustering       
        #self.discretizer = discretizer
        self.trajname = proteintraj
        
        # set basic directories
        self.dir_simulation_data = os.path.abspath(dir_simulation_data);
        self.dir_model = os.path.abspath(dir_model);
        
        # create sub-directories if they don't exist yet
        self.dir_model_crdlink = os.path.join(self.dir_model,"crdlink") 
        if (not os.path.isdir(self.dir_model_crdlink)):
            os.mkdir(self.dir_model_crdlink)
        self.dir_model_icrd = os.path.join(self.dir_model,"icrd") 
        if (not os.path.isdir(self.dir_model_icrd)):
            os.mkdir(self.dir_model_icrd)
        self.dir_model_dtraj = os.path.join(self.dir_model,"dtraj") 
        if (not os.path.isdir(self.dir_model_dtraj)):
            os.mkdir(self.dir_model_dtraj)        
        
        # filenames
        self.filename_lock = os.path.join(self.dir_model, settings.file_lock)


        """ file names of trajectories that have been analyzed already """
        self.received_results = [];
        self.transformed_trajectories = [];
        self.protein_trajectories = []; 
        
        # MSM building properties
        self.lag = lag;
        self.reversible = reversible;
        self.connected_set = connected_set;
        
        # Flags determining what to compute
        self.calculate_stationary = calculate_stationary;
        self.calculate_kinetics = calculate_kinetics;
        #print "calculate stationary = "+str(self.calculate_stationary);
        #print "calculate kinetics = "+str(self.calculate_kinetics);
        self.out = out;


    #############################################################################################################################            
    def __is_result(self, filepath):
        if not os.path.isdir(filepath):
            return False;
        for filename in os.listdir(filepath+'/'):
            if filename.endswith('.tdo'):
                return True;
        return False
        
    def __time(self):
        return strftime("%Y-%m-%d %H:%M:%S", gmtime())
    

    #############################################################################################################################            
    def __write_trajectory_list(self):
        """
        writes the list of the currently analyzed trajectories to disc
        """
        filename_out = os.path.join(self.dir_model, "trajectories.dat");
        fileout = open(filename_out, "w");
        for traj in self.protein_trajectories:
            fileout.write(os.path.abspath(traj)+"\n");
        fileout.close();
    
    
    #############################################################################################################################            
#     def __write_discretization(self):
#         """
#         writes the discretization definition to disc
#         """
#         # write trajectories
#         dtrajs = self.discretizer.get_discrete_trajectories();
#         for i in range(len(dtrajs)):
#             dtraj_path = os.path.join(self.dir_model_dtraj, self.received_results[i]+".dtraj");
#             dtrajout = open(dtraj_path, "w");
#             for s in dtrajs[i]:
#                 dtrajout.write(str(s)+"\n");
#             dtrajout.close();
#         # write discretization definition
#         disc = self.discretizer.get_discrete_state_definition(),
#         disc_path = os.path.join(self.dir_model, "discretization.dat");
#         discout = open(disc_path, "w");
#         for d in disc[0]:
#             discout.write(str(d[0])+"\t"+str(d[1])+"\n");
#         discout.close();
        


    def __read_dtraj(self):
        dtraj_files = os.listdir(self.dir_model_dtraj)
        dtrajs = []
        for f in dtraj_files:
            dtrajs.append(numpy.loadtxt(os.path.join(self.dir_model_dtraj,f), dtype=int))
        return dtrajs


    def __get_number_of_states(self):
        nstates = 0
        for dtraj in self.dtrajs:
            nstates = max(nstates, numpy.max(dtraj)+1)
        return nstates


    def __get_prior(self):
        return None
        #self.discretizer.get_prior()

    #############################################################################################################################            
    def __calculate_counts(self):
        # histogram
        self.dtrajs = self.__read_dtraj();
        self.nstates = self.__get_number_of_states();
        self.hist = numpy.bincount(numpy.concatenate(self.dtrajs), minlength=self.nstates);
        histout = open(os.path.join(self.dir_model, "histogram.dat"), "w");
        for i in range(len(self.hist)):
            histout.write(str(i)+"\t"+str(self.hist[i])+"\n");
        histout.close();
        
        # lag
        lagfile = open(os.path.join(self.dir_model, "lag.dat"), "w");
        lagfile.write(str(self.lag));
        lagfile.close();
        
        # Z
        self.Z = msm.count_matrix(self.dtrajs, lag=self.lag, nstates=self.nstates);
        msm.write_matrix_sparse(self.Z, os.path.join(self.dir_model, "Z.dat"));
        
        # connected component
        os.system("")
        # connected Z
        
        # full C
        self.C0 = self.__get_prior();
        if self.C0 == None:
            self.C = self.Z;
        else:
            self.C = self.C0 + self.Z;
        msm.write_matrix_sparse(self.C, os.path.join(self.dir_model, "C.dat"));


    #############################################################################################################################            
    def __calculate_stationary(self):
        if (not self.calculate_stationary):
            return;
        # T
        self.T = msm.transition_matrix(self.C);
        msm.write_matrix_sparse(self.T, os.path.join(self.dir_model, "T.dat"));
        
        # pi
        self.pi = msm.stationary_distribution(self.T);
        piout = open(os.path.join(self.dir_model, "pi.dat"), "w");
        for i in range(len(self.pi)):
            piout.write(str(i)+"\t"+str(self.pi[i])+"\n");
        piout.close();


    #############################################################################################################################            
    def __calculate_kinetics(self):
        if (not self.calculate_kinetics):
            return;
        # multi-lag its
        itsout = open(os.path.join(self.dir_model, "its.dat"), "w");
        for tau in [1,2,5,10,20,30,40,50,75,100,150,200,250,300,400,500]:
            Ztau = msm.count_matrix(self.dtrajs, lag=tau, nstates=self.nstates);
            if self.C0 == None:
                Ctau = Ztau;
            else:
                Ctau = self.C0 + Ztau;
            Ttau = msm.transition_matrix(Ctau);
            its = msm.timescales(Ttau, tau);
            itsout.write(str(tau)+"\t");
            for t in its:
                itsout.write(str(t)+"\t");
            itsout.write("\n")
        itsout.close();


    #############################################################################################################################            
    def __update_model(self):#, new_results):
#         if (len(new_results) > 0):
#             # put lockfile in model dir to show that the model is being updated
#             open(self.filename_lock, "w").close();
#             # ITERATE NEW RESULTS
#             for result in new_results:
#                 self.received_results.append(result);
#                 
#                 proteintraj = os.path.join(self.dir_simulation_data, result, self.trajname);
#                 
#                 # if protein trajectory really exists, append it to the trajectory list and do the transformation
#                 if os.path.isfile(proteintraj):
#                     self.protein_trajectories.append(proteintraj);
#                     
#                     # TRANSFORM
#                     if (self.transformer != None):
#                         transformedtraj = os.path.join(self.dir_model_icrd, result + ".icrd");
#                         # don't redo if the file already exists!
#                         if not os.path.isfile(transformedtraj): 
#                             self.out.write(self.__time()+"\ttransforming trajectory " + proteintraj + " -> " + transformedtraj + "\n");
#                             self.out.flush();       
#                             self.transformer.transform_trajectory(proteintraj, transformedtraj);
#                         # in any case, add trajectory to transformed trajectory list
#                         self.transformed_trajectories.append(transformedtraj);  
#                 else:
#                     self.out.write(self.__time()+"\tcould not find expected trajectory " + proteintraj + ". Ignoring.\n");
#                     self.out.flush();
#             
#             if (self.transformer != None):
#                 self.out.write(self.__time()+"\tfound " + str(len(self.transformed_trajectories)) + " transformed trajectories \n");    
#                 self.out.flush(); 
            
        # WRITE TRAJECTORY LIST
        self.__write_trajectory_list();
            
#             # CLUSTER
#             if (self.clustering != None):    
#                 self.out.write(self.__time()+"\clustering data ...\n");   
#                 if (self.transformer != None):
#                     intraj = self.transformed_trajectories
#                 else:
#                     intraj = self.protein_trajectories
#                 # cluster trajectories
#                 self.clustering.cluster_trajectories(self.dir_model, intraj);
#                 # new discretization of all trajectories required after clustering  
#                 self.out.write(self.__time()+"\discretizing trajectories ...\n");
#                 self.discretizer.discretize(self.dir_model, self.dir_model_dtraj, intraj);
#             else:
#                 # update discretization directly
#                 self.out.write(self.__time()+"\tupdating discretization ...\n");
#                 self.out.flush();       
#                 self.discretizer.discretize(self.transformed_trajectories);
#                 # write disc
#                 self.__write_discretization();

        # HISTOGRAM
        self.out.write(self.__time()+"\t Calculating counts\n");
        self.__calculate_counts();

        # STATIONARY DISTRIBUTION
        self.out.write(self.__time()+"\t Calculating stationary distribution\n");
        self.__calculate_stationary();

        # KINETICS
        self.out.write(self.__time()+"\t Calculating kinetics\n");
        self.__calculate_kinetics();

        # FINISH
        self.out.write(self.__time()+"\t... model is up to date!\n");
        self.out.flush();       


    #############################################################################################################################            
    def update(self, selected_results = None):
        """
        look through results directory, find new trajectories, and process them
        """
        if (selected_results == None):
            selected_results = os.listdir(self.dir_simulation_data);
            
        new_results = []
        for filename in selected_results:
            filepath = os.path.join(self.dir_simulation_data, filename)
            if self.__is_result(filepath):
                if not filename in self.received_results:            
                    new_results.append(filename);
        
        if new_results:
            # lock directory
            open(self.filename_lock, "w").close();
            
            self.out.write(self.__time()+"\treceived " + str(len(new_results)) + " results " + str(new_results) + " \n");
            self.out.flush();
            # add symbolic links
            for result in new_results:
                proteintraj = os.path.join(self.dir_simulation_data, result, self.trajname);
                proteinname, proteinext = os.path.splitext(self.trajname)
                linkname = os.path.join(self.dir_model_crdlink, result+proteinext)
                if (not os.path.exists(linkname)):
                    os.symlink(proteintraj, linkname)
            # apply transformations
            for t in self.transformers:
                t.update();
            # update model
            self.__update_model()
            
            # release directory
            if (os.path.exists(self.filename_lock)):
                os.remove(self.filename_lock);
            
            #self.__process_new_results(new_results) 
        # return number of new results
        numresults = len(new_results)
        return(numresults);


    #############################################################################################################################            
    def get_received_results(self):
        return self.received_results;

