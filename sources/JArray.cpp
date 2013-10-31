/*
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#ifdef PYTHON

#include <jni.h>
#include <Python.h>
#include "structmember.h"

#include "JArray.h"
#include "functions.h"
#include "java/lang/Class.h"

using namespace java::lang;


template<typename T> class _t_JArray : public t_JArray<T> {
public:
    static PyObject *format;
};

template<typename U>
static PyObject *get(U *self, Py_ssize_t n)
{
    return self->array.get(n);
}

template<typename U>
static PyObject *toSequence(U *self)
{
    return self->array.toSequence();
}

template<typename U>
static PyObject *toSequence(U *self, Py_ssize_t lo, Py_ssize_t hi)
{
    return self->array.toSequence(lo, hi);
}

template<typename U> class _t_iterator {
public:
    PyObject_HEAD
    U *obj;
    Py_ssize_t position;

    static void dealloc(_t_iterator *self)
    {
        Py_XDECREF(self->obj);
        self->ob_type->tp_free((PyObject *) self);
    }

    static PyObject *iternext(_t_iterator *self)
    {
        if (self->position < (Py_ssize_t) self->obj->array.length)
            return get<U>(self->obj, self->position++);

        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    static PyTypeObject *JArrayIterator;
};

template<typename T, typename U>
static int init(U *self, PyObject *args, PyObject *kwds)
{
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O", &obj))
        return -1;

    if (PySequence_Check(obj))
    {
        self->array = JArray<T>(obj);
        if (PyErr_Occurred())
            return -1;
    }
    else if (PyGen_Check(obj))
    {
        PyObject *tuple =
            PyObject_CallFunctionObjArgs((PyObject *) &PyTuple_Type, obj, NULL);

        if (!tuple)
            return -1;

        self->array = JArray<T>(tuple);
        Py_DECREF(tuple);
        if (PyErr_Occurred())
            return -1;
    }
    else if (PyInt_Check(obj))
    {
        int n = PyInt_AsLong(obj);

        if (n < 0)
        {
            PyErr_SetObject(PyExc_ValueError, obj);
            return -1;
        }

        self->array = JArray<T>(n);
    }
    else
    {
        PyErr_SetObject(PyExc_TypeError, obj);
        return -1;
    }

    return 0;
}

template<typename T, typename U>
static void dealloc(U *self)
{
    self->array = JArray<T>((jobject) NULL);
    self->ob_type->tp_free((PyObject *) self);
}

template<typename U>
static PyObject *_format(U *self, PyObject *(*fn)(PyObject *))
{
    if (self->array.this$)
    {
        PyObject *list = toSequence<U>(self);
            
        if (list)
        {
            PyObject *result = (*fn)(list);

            Py_DECREF(list);
            if (result)
            {
                PyObject *args = PyTuple_New(1);

                PyTuple_SET_ITEM(args, 0, result);
                result = PyString_Format(U::format, args);
                Py_DECREF(args);

                return result;
            }
        }

        return NULL;
    }

    return PyString_FromString("<null>");
}

template<typename U>
static PyObject *repr(U *self)
{
    return _format(self, (PyObject *(*)(PyObject *)) PyObject_Repr);
}

template<typename U>
static PyObject *str(U *self)
{
    return _format(self, (PyObject *(*)(PyObject *)) PyObject_Str);
}

template<typename U>
static int _compare(U *self, PyObject *value, int i0, int i1, int op, int *cmp)
{
    PyObject *v0 = get<U>(self, i0);
    PyObject *v1 = PySequence_Fast_GET_ITEM(value, i1);  /* borrowed */

    if (!v0)
        return -1;

    if (!v1)
    {
        Py_DECREF(v0);
        return -1;
    }

    *cmp = PyObject_RichCompareBool(v0, v1, op);
    Py_DECREF(v0);

    if (*cmp < 0)
        return -1;

    return 0;
}

template<typename U>
static PyObject *richcompare(U *self, PyObject *value, int op)
{
    PyObject *result = NULL;
    int s0, s1;

    if (!PySequence_Check(value))
    {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    value = PySequence_Fast(value, "not a sequence");
    if (!value)
        return NULL;

    s0 = PySequence_Fast_GET_SIZE(value);
    s1 = self->array.length;

    if (s1 < 0)
    {
        Py_DECREF(value);
        return NULL;
    }

    if (s0 != s1)
    {
        switch (op) {
          case Py_EQ: result = Py_False; break;
          case Py_NE: result = Py_True; break;
        }
    }

    if (!result)
    {
        int i0, i1, cmp = 1;

        for (i0 = 0, i1 = 0; i0 < s0 && i1 < s1 && cmp; i0++, i1++) {
            if (_compare(self, value, i0, i1, Py_EQ, &cmp) < 0)
            {
                Py_DECREF(value);
                return NULL;
            }                
        }

        if (cmp)
        {
            switch (op) {
              case Py_LT: cmp = s0 < s1; break;
              case Py_LE: cmp = s0 <= s1; break;
              case Py_EQ: cmp = s0 == s1; break;
              case Py_NE: cmp = s0 != s1; break;
              case Py_GT: cmp = s0 > s1; break;
              case Py_GE: cmp = s0 >= s1; break;
              default: cmp = 0;
            }

            result = cmp ? Py_True : Py_False;
        }
        else if (op == Py_EQ)
            result = Py_False;
        else if (op == Py_NE)
            result = Py_True;
        else if (_compare(self, value, i0, i1, op, &cmp) < 0)
        {
            Py_DECREF(value);
            return NULL;
        }
        else
            result = cmp ? Py_True : Py_False;
    }
    Py_DECREF(value);

    Py_INCREF(result);
    return result;
}

template<typename U>
static PyObject *iter(U *self)
{
    _t_iterator<U> *it =
        PyObject_New(_t_iterator<U>, _t_iterator<U>::JArrayIterator);

    if (it)
    {
        it->position = 0;
        it->obj = self; Py_INCREF((PyObject *) self);
    }

    return (PyObject *) it;
}

template<typename U>
static Py_ssize_t seq_length(U *self)
{
    if (self->array.this$)
        return self->array.length;

    return 0;
}

template<typename U>
static PyObject *seq_get(U *self, Py_ssize_t n)
{
    return get<U>(self, n);
}

template<typename U>
static int seq_contains(U *self, PyObject *value)
{
    return 0;
}

template<typename U>
static PyObject *seq_concat(U *self, PyObject *arg)
{
    PyObject *list = toSequence<U>(self);

    if (list != NULL &&
        PyList_Type.tp_as_sequence->sq_inplace_concat(list, arg) < 0)
    {
        Py_DECREF(list);
        return NULL;
    }

    return list;
}

template<typename U>
static PyObject *seq_repeat(U *self, Py_ssize_t n)
{
    PyObject *list = toSequence<U>(self);

    if (list != NULL &&
        PyList_Type.tp_as_sequence->sq_inplace_repeat(list, n) < 0)
    {
        Py_DECREF(list);
        return NULL;
    }

    return list;
}

template<typename U>
static PyObject *seq_getslice(U *self, Py_ssize_t lo, Py_ssize_t hi)
{
    return toSequence<U>(self, lo, hi);
}

template<typename U>
static int seq_set(U *self, Py_ssize_t n, PyObject *value)
{
    return self->array.set(n, value);
}

template<typename U>
static int seq_setslice(U *self, Py_ssize_t lo, Py_ssize_t hi, PyObject *values)
{
    Py_ssize_t length = self->array.length;

    if (values == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "array size cannot change");
        return -1;
    }
            
    if (lo < 0) lo = length + lo;
    if (lo < 0) lo = 0;
    else if (lo > length) lo = length;
    if (hi < 0) hi = length + hi;
    if (hi < 0) hi = 0;
    else if (hi > length) hi = length;
    if (lo > hi) lo = hi;

    PyObject *sequence = PySequence_Fast(values, "not a sequence");
    if (!sequence)
        return -1;

    Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    if (size < 0)
        goto error;

    if (size != hi - lo)
    {
        PyErr_SetString(PyExc_ValueError, "array size cannot change");
        goto error;
    }

    for (Py_ssize_t i = lo; i < hi; i++) {
        PyObject *value = PySequence_Fast_GET_ITEM(sequence, i - lo);

        if (value == NULL)
            goto error;

        if (self->array.set(i, value) < 0)
            goto error;
    }

    Py_DECREF(sequence);
    return 0;

  error:
    Py_DECREF(sequence);
    return -1;
}

template<typename T> 
static jclass initializeClass(bool getOnly)
{
    return env->get_vm_env()->GetObjectClass(JArray<T>((Py_ssize_t) 0).this$);
}

template<typename T> 
static PyObject *cast_(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *arg, *clsObj;

    if (!PyArg_ParseTuple(args, "O", &arg))
        return NULL;

    if (!PyObject_TypeCheck(arg, &PY_TYPE(Object)))
    {
        PyErr_SetObject(PyExc_TypeError, arg);
        return NULL;
    }

    Class argCls = ((t_Object *) arg)->object.getClass();

    if (!argCls.isArray())
    {
        PyErr_SetObject(PyExc_TypeError, arg);
        return NULL;
    }

    clsObj = PyObject_GetAttrString((PyObject *) type, "class_");
    if (!clsObj)
        return NULL;

    Class arrayCls = ((t_Class *) clsObj)->object;

    if (!arrayCls.isAssignableFrom(argCls))
    {
        PyErr_SetObject(PyExc_TypeError, arg);
        return NULL;
    }

    return JArray<T>(((t_JObject *) arg)->object.this$).wrap();
}

template<typename T> 
static PyObject *instance_(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *arg, *clsObj;

    if (!PyArg_ParseTuple(args, "O", &arg))
        return NULL;

    if (!PyObject_TypeCheck(arg, &PY_TYPE(Object)))
        Py_RETURN_FALSE;

    Class argCls = ((t_Object *) arg)->object.getClass();

    if (!argCls.isArray())
        Py_RETURN_FALSE;

    clsObj = PyObject_GetAttrString((PyObject *) type, "class_");
    if (!clsObj)
        return NULL;

    Class arrayCls = ((t_Class *) clsObj)->object;

    if (!arrayCls.isAssignableFrom(argCls))
        Py_RETURN_FALSE;

    Py_RETURN_TRUE;
}

template<typename T> 
static PyObject *assignable_(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    return instance_<T>(type, args, kwds);
}

template< typename T, typename U = _t_JArray<T> > class jarray_type {
public:
    PySequenceMethods seq_methods;
    PyTypeObject type_object;

    class iterator_type {
    public:
        PyTypeObject type_object;

        void install(char *name, PyObject *module)
        {
            type_object.tp_name = name;

            if (PyType_Ready(&type_object) == 0)
            {
                Py_INCREF((PyObject *) &type_object);
                PyModule_AddObject(module, name, (PyObject *) &type_object);
            }

            _t_iterator<U>::JArrayIterator = &type_object;
        }

        iterator_type()
        {
            memset(&type_object, 0, sizeof(type_object));

            type_object.ob_refcnt = 1;
            type_object.ob_type = NULL;
            type_object.tp_basicsize = sizeof(_t_iterator<U>);
            type_object.tp_dealloc = (destructor) _t_iterator<U>::dealloc;
            type_object.tp_flags = Py_TPFLAGS_DEFAULT;
            type_object.tp_doc = "JArrayIterator<T> wrapper type";
            type_object.tp_iter = (getiterfunc) PyObject_SelfIter;
            type_object.tp_iternext = (iternextfunc) _t_iterator<U>::iternext;
        }
    };

    iterator_type iterator_type_object;

    void install(char *name, char *type_name, char *iterator_name,
                 PyObject *module)
    {
        type_object.tp_name = name;

        if (PyType_Ready(&type_object) == 0)
        {
            Py_INCREF((PyObject *) &type_object);
            PyDict_SetItemString(type_object.tp_dict, "class_",
                                 make_descriptor(initializeClass<T>));
            
            PyModule_AddObject(module, name, (PyObject *) &type_object);
        }

        U::format = PyString_FromFormat("JArray<%s>%%s", type_name);
        iterator_type_object.install(iterator_name, module);
    }

    static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {
        U *self = (U *) type->tp_alloc(type, 0);

        if (self)
            self->array = JArray<T>((jobject) NULL);

        return (PyObject *) self;
    }

    jarray_type()
    {
        memset(&seq_methods, 0, sizeof(seq_methods));
        memset(&type_object, 0, sizeof(type_object));

        static PyMethodDef methods[] = {
            { "cast_",
              (PyCFunction) (PyObject *(*)(PyTypeObject *,
                                           PyObject *, PyObject *))
              cast_<T>,
              METH_VARARGS | METH_CLASS, NULL },
            { "instance_",
              (PyCFunction) (PyObject *(*)(PyTypeObject *,
                                           PyObject *, PyObject *))
              instance_<T>,
              METH_VARARGS | METH_CLASS, NULL },
            { "assignable_",
              (PyCFunction) (PyObject *(*)(PyTypeObject *,
                                           PyObject *, PyObject *))
              assignable_<T>,
              METH_VARARGS | METH_CLASS, NULL },
            { NULL, NULL, 0, NULL }
        };

        seq_methods.sq_length =
            (lenfunc) (Py_ssize_t (*)(U *)) seq_length<U>;
        seq_methods.sq_concat =
            (binaryfunc) (PyObject *(*)(U *, PyObject *)) seq_concat<U>;
        seq_methods.sq_repeat =
            (ssizeargfunc) (PyObject *(*)(U *, Py_ssize_t)) seq_repeat<U>;
        seq_methods.sq_item =
            (ssizeargfunc) (PyObject *(*)(U *, Py_ssize_t)) seq_get<U>;
        seq_methods.sq_slice =
            (ssizessizeargfunc) (PyObject *(*)(U *, Py_ssize_t, Py_ssize_t))
            seq_getslice<U>;
        seq_methods.sq_ass_item =
            (ssizeobjargproc) (int (*)(U *, Py_ssize_t, PyObject *)) seq_set<U>;
        seq_methods.sq_ass_slice =
            (ssizessizeobjargproc) (int (*)(U *, Py_ssize_t, Py_ssize_t,
                                            PyObject *)) seq_setslice<U>;
        seq_methods.sq_contains =
            (objobjproc) (int (*)(U *, PyObject *)) seq_contains<U>;
        seq_methods.sq_inplace_concat = NULL;
        seq_methods.sq_inplace_repeat = NULL;

        type_object.ob_refcnt = 1;
        type_object.tp_basicsize = sizeof(U);
        type_object.tp_dealloc = (destructor) (void (*)(U *)) dealloc<T,U>;
        type_object.tp_repr = (reprfunc) (PyObject *(*)(U *)) repr<U>;
        type_object.tp_as_sequence = &seq_methods;
        type_object.tp_str = (reprfunc) (PyObject *(*)(U *)) str<U>;
        type_object.tp_flags = Py_TPFLAGS_DEFAULT;
        type_object.tp_doc = "JArray<T> wrapper type";
        type_object.tp_richcompare =
            (richcmpfunc) (PyObject *(*)(U *, PyObject *, int)) richcompare<U>;
        type_object.tp_iter = (getiterfunc) (PyObject *(*)(U *)) iter<U>;
        type_object.tp_methods = methods;
        type_object.tp_base = &PY_TYPE(Object);
        type_object.tp_init =
            (initproc) (int (*)(U *, PyObject *, PyObject *)) init<T,U>;
        type_object.tp_new = (newfunc) _new;
    }
};

template<typename T> class _t_jobjectarray : public _t_JArray<T> {
public:
    PyObject *(*wrapfn)(const T&);
};

template<> PyObject *get(_t_jobjectarray<jobject> *self, Py_ssize_t n)
{
    return self->array.get(n, self->wrapfn);
}

template<> PyObject *toSequence(_t_jobjectarray<jobject> *self)
{
    return self->array.toSequence(self->wrapfn);
}

template<> PyObject *toSequence(_t_jobjectarray<jobject> *self,
                                Py_ssize_t lo, Py_ssize_t hi)
{
    return self->array.toSequence(lo, hi, self->wrapfn);
}

template<> int init< jobject,_t_jobjectarray<jobject> >(_t_jobjectarray<jobject> *self, PyObject *args, PyObject *kwds)
{
    PyObject *obj, *clsObj = NULL;
    PyObject *(*wrapfn)(const jobject &) = NULL;
    jclass cls;

    if (!PyArg_ParseTuple(args, "O|O", &obj, &clsObj))
        return -1;

    if (clsObj == NULL)
        cls = env->findClass("java/lang/Object");
    else if (PyObject_TypeCheck(clsObj, &PY_TYPE(Class)))
        cls = (jclass) ((t_Class *) clsObj)->object.this$;
    else if (PyType_Check(clsObj))
    {
        if (PyType_IsSubtype((PyTypeObject *) clsObj, &PY_TYPE(JObject)))
        {
            PyObject *cobj = PyObject_GetAttrString(clsObj, "wrapfn_");

            if (cobj == NULL)
                PyErr_Clear();
            else
            {
                wrapfn = (PyObject *(*)(const jobject &))
                    PyCObject_AsVoidPtr(cobj);
                Py_DECREF(cobj);
            }

            clsObj = PyObject_GetAttrString(clsObj, "class_");
            if (clsObj == NULL)
                return -1;

            cls = (jclass) ((t_Class *) clsObj)->object.this$;
            Py_DECREF(clsObj);
        }
        else
        {
            PyErr_SetObject(PyExc_ValueError, clsObj);
            return -1;
        }
    }
    else
    {
        PyErr_SetObject(PyExc_TypeError, clsObj);
        return -1;
    }

    if (PySequence_Check(obj))
    {
        self->array = JArray<jobject>(cls, obj);
        if (PyErr_Occurred())
            return -1;
    }
    else if (PyGen_Check(obj))
    {
        PyObject *tuple =
            PyObject_CallFunctionObjArgs((PyObject *) &PyTuple_Type, obj, NULL);

        if (!tuple)
            return -1;

        self->array = JArray<jobject>(cls, tuple);
        Py_DECREF(tuple);
        if (PyErr_Occurred())
            return -1;
    }
    else if (PyInt_Check(obj))
    {
        int n = PyInt_AsLong(obj);

        if (n < 0)
        {
            PyErr_SetObject(PyExc_ValueError, obj);
            return -1;
        }

        self->array = JArray<jobject>(cls, n);
    }
    else
    {
        PyErr_SetObject(PyExc_TypeError, obj);
        return -1;
    }

    self->wrapfn = wrapfn;

    return 0;
}

template<> jclass initializeClass<jobject>(bool getOnly)
{
    jclass cls = env->findClass("java/lang/Object");
    return env->get_vm_env()->GetObjectClass(JArray<jobject>(cls, (Py_ssize_t) 0).this$);
}

template<> PyObject *cast_<jobject>(PyTypeObject *type,
				    PyObject *args, PyObject *kwds)
{
    PyObject *arg, *clsArg = NULL;
    PyObject *(*wrapfn)(const jobject&) = NULL;
    jclass elementCls;

    if (!PyArg_ParseTuple(args, "O|O", &arg, &clsArg))
        return NULL;

    if (!PyObject_TypeCheck(arg, &PY_TYPE(Object)))
    {
        PyErr_SetObject(PyExc_TypeError, arg);
        return NULL;
    }

    Class argCls = ((t_Object *) arg)->object.getClass();

    if (!argCls.isArray())
    {
        PyErr_SetObject(PyExc_TypeError, arg);
        return NULL;
    }

    if (clsArg != NULL)
    {
        if (!PyType_Check(clsArg))
        {
            PyErr_SetObject(PyExc_TypeError, clsArg);
            return NULL;
        }
        else if (!PyType_IsSubtype((PyTypeObject *) clsArg, &PY_TYPE(JObject)))
        {
            PyErr_SetObject(PyExc_ValueError, clsArg);
            return NULL;
        }

        PyObject *cobj = PyObject_GetAttrString(clsArg, "wrapfn_");

        if (cobj == NULL)
            PyErr_Clear();
        else
        {
            wrapfn = (PyObject *(*)(const jobject &)) PyCObject_AsVoidPtr(cobj);
            Py_DECREF(cobj);
        }

        clsArg = PyObject_GetAttrString(clsArg, "class_");
        if (clsArg == NULL)
            return NULL;

        elementCls = (jclass) ((t_Class *) clsArg)->object.this$;
        Py_DECREF(clsArg);
    }
    else
        elementCls = env->findClass("java/lang/Object");

    JNIEnv *vm_env = env->get_vm_env();
    jobjectArray array = vm_env->NewObjectArray(0, elementCls, NULL);
    Class arrayCls(vm_env->GetObjectClass((jobject) array));

    if (!arrayCls.isAssignableFrom(argCls))
    {
        PyErr_SetObject(PyExc_TypeError, arg);
        return NULL;
    }

    return JArray<jobject>(((t_JObject *) arg)->object.this$).wrap(wrapfn);
}

template<> PyObject *instance_<jobject>(PyTypeObject *type,
					PyObject *args, PyObject *kwds)
{
    PyObject *arg, *clsArg = NULL;
    jclass elementCls;

    if (!PyArg_ParseTuple(args, "O|O", &arg, &clsArg))
        return NULL;

    if (!PyObject_TypeCheck(arg, &PY_TYPE(Object)))
        Py_RETURN_FALSE;

    Class argCls = ((t_Object *) arg)->object.getClass();

    if (!argCls.isArray())
        Py_RETURN_FALSE;

    if (clsArg != NULL)
    {
        if (!PyType_Check(clsArg))
        {
            PyErr_SetObject(PyExc_TypeError, clsArg);
            return NULL;
        }
        else if (!PyType_IsSubtype((PyTypeObject *) clsArg, &PY_TYPE(JObject)))
        {
            PyErr_SetObject(PyExc_ValueError, clsArg);
            return NULL;
        }

        clsArg = PyObject_GetAttrString(clsArg, "class_");
        if (clsArg == NULL)
            return NULL;

        elementCls = (jclass) ((t_Class *) clsArg)->object.this$;
        Py_DECREF(clsArg);
    }
    else
        elementCls = env->findClass("java/lang/Object");

    JNIEnv *vm_env = env->get_vm_env();
    jobjectArray array = vm_env->NewObjectArray(0, elementCls, NULL);
    Class arrayCls(vm_env->GetObjectClass((jobject) array));

    if (!arrayCls.isAssignableFrom(argCls))
        Py_RETURN_FALSE;

    Py_RETURN_TRUE;
}

template<> PyObject *assignable_<jobject>(PyTypeObject *type,
					  PyObject *args, PyObject *kwds)
{
    PyObject *arg, *clsArg = NULL;
    jclass elementCls;

    if (!PyArg_ParseTuple(args, "O|O", &arg, &clsArg))
        return NULL;

    if (!PyObject_TypeCheck(arg, &PY_TYPE(Object)))
        Py_RETURN_FALSE;

    Class argCls = ((t_Object *) arg)->object.getClass();

    if (!argCls.isArray())
        Py_RETURN_FALSE;

    if (clsArg != NULL)
    {
        if (!PyType_Check(clsArg))
        {
            PyErr_SetObject(PyExc_TypeError, clsArg);
            return NULL;
        }
        else if (!PyType_IsSubtype((PyTypeObject *) clsArg, &PY_TYPE(JObject)))
        {
            PyErr_SetObject(PyExc_ValueError, clsArg);
            return NULL;
        }

        clsArg = PyObject_GetAttrString(clsArg, "class_");
        if (clsArg == NULL)
            return NULL;

        elementCls = (jclass) ((t_Class *) clsArg)->object.this$;
        Py_DECREF(clsArg);
    }
    else
        elementCls = env->findClass("java/lang/Object");

    JNIEnv *vm_env = env->get_vm_env();
    jobjectArray array = vm_env->NewObjectArray(0, elementCls, NULL);
    Class arrayCls(vm_env->GetObjectClass((jobject) array));

    if (!argCls.isAssignableFrom(arrayCls))
        Py_RETURN_FALSE;

    Py_RETURN_TRUE;
}


template<typename T> PyTypeObject *_t_iterator<T>::JArrayIterator;
template<typename T> PyObject *_t_JArray<T>::format;

static jarray_type< jobject, _t_jobjectarray<jobject> > jarray_jobject;

static jarray_type<jstring> jarray_jstring;
static jarray_type<jboolean> jarray_jboolean;
static jarray_type<jbyte> jarray_jbyte;
static jarray_type<jchar> jarray_jchar;
static jarray_type<jdouble> jarray_jdouble;
static jarray_type<jfloat> jarray_jfloat;
static jarray_type<jint> jarray_jint;
static jarray_type<jlong> jarray_jlong;
static jarray_type<jshort> jarray_jshort;


PyObject *JArray<jobject>::wrap(PyObject *(*wrapfn)(const jobject&)) const
{
    if (this$ != NULL)
    {
        _t_jobjectarray<jobject> *obj =
            PyObject_New(_t_jobjectarray<jobject>, &jarray_jobject.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jobject>));
        obj->array = *this;
        obj->wrapfn = wrapfn;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jstring>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jstring> *obj =
            PyObject_New(_t_JArray<jstring>, &jarray_jstring.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jstring>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jboolean>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jboolean> *obj =
            PyObject_New(_t_JArray<jboolean>, &jarray_jboolean.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jboolean>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jbyte>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jbyte> *obj =
            PyObject_New(_t_JArray<jbyte>, &jarray_jbyte.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jbyte>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jchar>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jchar> *obj =
            PyObject_New(_t_JArray<jchar>, &jarray_jchar.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jchar>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jdouble>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jdouble> *obj =
            PyObject_New(_t_JArray<jdouble>, &jarray_jdouble.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jdouble>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jfloat>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jfloat> *obj =
            PyObject_New(_t_JArray<jfloat>, &jarray_jfloat.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jfloat>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jint>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jint> *obj =
            PyObject_New(_t_JArray<jint>, &jarray_jint.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jint>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jlong>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jlong> *obj =
            PyObject_New(_t_JArray<jlong>, &jarray_jlong.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jlong>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray<jshort>::wrap() const
{
    if (this$ != NULL)
    {
        _t_JArray<jshort> *obj =
            PyObject_New(_t_JArray<jshort>, &jarray_jshort.type_object);

        memset(&(obj->array), 0, sizeof(JArray<jshort>));
        obj->array = *this;

        return (PyObject *) obj;
    }

    Py_RETURN_NONE;
}

PyObject *JArray_Type(PyObject *self, PyObject *arg)
{
    PyObject *type_name = NULL, *type;
    char const *name = NULL;

    if (PyType_Check(arg))
    {
        type_name = PyObject_GetAttrString(arg, "__name__");
        if (!type_name)
            return NULL;
    }
    else if (PyString_Check(arg))
    {
        type_name = arg;
        Py_INCREF(type_name);
    }
    else if (PyFloat_Check(arg))
    {
        type_name = NULL;
        name = "double";
    }
    else
    {
        PyObject *arg_type = (PyObject *) arg->ob_type;

        type_name = PyObject_GetAttrString(arg_type, "__name__");
        if (!type_name)
            return NULL;
    }

    if (type_name != NULL)
    {
        name = PyString_AsString(type_name);
        if (!name)
        {
            Py_DECREF(type_name);
            return NULL;
        }
    }

    if (!strcmp(name, "object"))
        type = (PyObject *) &jarray_jobject.type_object;
    else if (!strcmp(name, "string"))
        type = (PyObject *) &jarray_jstring.type_object;
    else if (!strcmp(name, "bool"))
        type = (PyObject *) &jarray_jboolean.type_object;
    else if (!strcmp(name, "byte"))
        type = (PyObject *) &jarray_jbyte.type_object;
    else if (!strcmp(name, "char"))
        type = (PyObject *) &jarray_jchar.type_object;
    else if (!strcmp(name, "double"))
        type = (PyObject *) &jarray_jdouble.type_object;
    else if (!strcmp(name, "float"))
        type = (PyObject *) &jarray_jfloat.type_object;
    else if (!strcmp(name, "int"))
        type = (PyObject *) &jarray_jint.type_object;
    else if (!strcmp(name, "long"))
        type = (PyObject *) &jarray_jlong.type_object;
    else if (!strcmp(name, "short"))
        type = (PyObject *) &jarray_jshort.type_object;
    else
    {
        PyErr_SetObject(PyExc_ValueError, arg);
        Py_XDECREF(type_name);

        return NULL;
    }

    Py_INCREF(type);
    Py_XDECREF(type_name);

    return type;
}

static PyObject *t_JArray_jbyte__get_string_(t_JArray<jbyte> *self, void *data)
{
    return self->array.to_string_();
}

static PyGetSetDef t_JArray_jbyte__fields[] = {
    { "string_", (getter) t_JArray_jbyte__get_string_, NULL, "", NULL },
    { NULL, NULL, NULL, NULL, NULL }
};


PyTypeObject *PY_TYPE(JArrayObject);
PyTypeObject *PY_TYPE(JArrayString);
PyTypeObject *PY_TYPE(JArrayBool);
PyTypeObject *PY_TYPE(JArrayByte);
PyTypeObject *PY_TYPE(JArrayChar);
PyTypeObject *PY_TYPE(JArrayDouble);
PyTypeObject *PY_TYPE(JArrayFloat);
PyTypeObject *PY_TYPE(JArrayInt);
PyTypeObject *PY_TYPE(JArrayLong);
PyTypeObject *PY_TYPE(JArrayShort);


void _install_jarray(PyObject *module)
{
    jarray_jobject.install("JArray_object", "object",
                            "__JArray_object_iterator", module);
    PY_TYPE(JArrayObject) = &jarray_jobject.type_object;

    jarray_jstring.install("JArray_string", "string",
                            "__JArray_string_iterator", module);
    PY_TYPE(JArrayString) = &jarray_jstring.type_object;

    jarray_jboolean.install("JArray_bool", "bool",
                            "__JArray_bool_iterator", module);
    PY_TYPE(JArrayBool) = &jarray_jboolean.type_object;

    jarray_jbyte.type_object.tp_getset = t_JArray_jbyte__fields;
    jarray_jbyte.install("JArray_byte", "byte",
                         "__JArray_byte_iterator", module);
    PY_TYPE(JArrayByte) = &jarray_jbyte.type_object;

    jarray_jchar.install("JArray_char", "char",
                         "__JArray_char_iterator", module);
    PY_TYPE(JArrayChar) = &jarray_jchar.type_object;

    jarray_jdouble.install("JArray_double", "double",
                           "__JArray_double_iterator", module);
    PY_TYPE(JArrayDouble) = &jarray_jdouble.type_object;

    jarray_jfloat.install("JArray_float", "float",
                          "__JArray_float_iterator", module);
    PY_TYPE(JArrayFloat) = &jarray_jfloat.type_object;

    jarray_jint.install("JArray_int", "int",
                        "__JArray_int_iterator", module);
    PY_TYPE(JArrayInt) = &jarray_jint.type_object;

    jarray_jlong.install("JArray_long", "long",
                         "__JArray_long_iterator", module);
    PY_TYPE(JArrayLong) = &jarray_jlong.type_object;

    jarray_jshort.install("JArray_short", "short",
                          "__JArray_short_iterator", module);
    PY_TYPE(JArrayShort) = &jarray_jshort.type_object;
}

#endif /* PYTHON */
