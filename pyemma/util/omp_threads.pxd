cdef extern from "omp.h":
  void omp_set_num_threads (int)
  int omp_get_num_threads () 
  int omp_get_max_threads () 
  int omp_get_thread_num () 
  int omp_get_num_procs () 

  int omp_in_parallel ()

  void omp_set_dynamic (int)
  int omp_get_dynamic ()
 

