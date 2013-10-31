from numpy import ndarray

__all__ = ['ArrayWrapper']

class ArrayWrapper(ndarray):
    """
    This subclass of numpy multidimensional array class aims to wrap array types
    from the Stallone library for easy mathematical operations.
    
    Currently it copies the memory, because the Python Java wrapper for arrays
    JArray<T> does not suggerate continous memory layout, which is needed for
    direct wrapping.
    """
    def __new__(cls, *args, **kwargs):
        import stallone as st
        from numpy import float64, int32, int64, array as nparray, empty
        # if first argument is of type IIntArray or IDoubleArray
        if isinstance(args[0], (st.IIntArray, st.IDoubleArray)):
            dtype = None
            caster = None
    
            from platform import architecture
            arch = architecture()[0]
            if type(args[0]) == st.IDoubleArray:
                dtype = float64
                caster = st.JArray_double.cast_
            elif type(args[0]) == st.IIntArray:
                caster = st.JArray_int.cast_
                if arch == '64bit':  
                    dtype = int64 # long int?
                else:
                    dtype = int32

            d_arr = args[0]
            rows = d_arr.rows()
            cols = d_arr.columns()
            order = d_arr.order() 
            size = d_arr.size()
            
            isSparse = d_arr.isSparse()
            
            if order < 2:
                #arr =  np.fromiter(d_arr.getArray(), dtype=dtype)
                # np.frombuffer(d_arr.getArray(), dtype=dtype, count=size )
                arr = nparray(d_arr.getArray(), dtype=dtype)
            elif order == 2:
                table = d_arr.getTable()
                arr = empty((rows, cols))
                # assign rows
                for i in xrange(rows):
                    jarray = caster(table[i])
                    row = nparray(jarray, dtype=dtype)
                    arr[i] = row
            elif order == 3:
                raise NotImplemented
                
            arr.shape = (rows, cols)
            return arr
            
        return ndarray.__new__(cls, *args, **kwargs)
