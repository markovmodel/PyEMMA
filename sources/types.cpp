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

#include <jni.h>
#include <Python.h>
#include "structmember.h"

#include "java/lang/Object.h"
#include "java/lang/Class.h"
#include "functions.h"

using namespace java::lang;


/* FinalizerProxy */

static PyObject *t_fc_call(PyObject *self, PyObject *args, PyObject *kwds);

static void t_fp_dealloc(t_fp *self);
static PyObject *t_fp_getattro(t_fp *self, PyObject *name);
static int t_fp_setattro(t_fp *self, PyObject *name, PyObject *value);
static int t_fp_traverse(t_fp *self, visitproc visit, void *arg);
static int t_fp_clear(t_fp *self);
static PyObject *t_fp_repr(t_fp *self);
static PyObject *t_fp_iter(t_fp *self);

static Py_ssize_t t_fp_map_length(t_fp *self);
static PyObject *t_fp_map_get(t_fp *self, PyObject *key);
static int t_fp_map_set(t_fp *self, PyObject *key, PyObject *value);

static Py_ssize_t t_fp_seq_length(t_fp *self);
static PyObject *t_fp_seq_get(t_fp *self, Py_ssize_t n);
static int t_fp_seq_contains(t_fp *self, PyObject *value);
static PyObject *t_fp_seq_concat(t_fp *self, PyObject *arg);
static PyObject *t_fp_seq_repeat(t_fp *self, Py_ssize_t n);
static PyObject *t_fp_seq_getslice(t_fp *self, Py_ssize_t low, Py_ssize_t high);
static int t_fp_seq_set(t_fp *self, Py_ssize_t i, PyObject *value);
static int t_fp_seq_setslice(t_fp *self, Py_ssize_t low,
                             Py_ssize_t high, PyObject *arg);
static PyObject *t_fp_seq_inplace_concat(t_fp *self, PyObject *arg);
static PyObject *t_fp_seq_inplace_repeat(t_fp *self, Py_ssize_t n);


PyTypeObject PY_TYPE(FinalizerClass) = {
    PyObject_HEAD_INIT(NULL)
    0,                                   /* ob_size */
    "jcc.FinalizerClass",                /* tp_name */
    PyType_Type.tp_basicsize,            /* tp_basicsize */
    0,                                   /* tp_itemsize */
    0,                                   /* tp_dealloc */
    0,                                   /* tp_print */
    0,                                   /* tp_getattr */
    0,                                   /* tp_setattr */
    0,                                   /* tp_compare */
    0,                                   /* tp_repr */
    0,                                   /* tp_as_number */
    0,                                   /* tp_as_sequence */
    0,                                   /* tp_as_mapping */
    0,                                   /* tp_hash  */
    (ternaryfunc) t_fc_call,             /* tp_call */
    0,                                   /* tp_str */
    0,                                   /* tp_getattro */
    0,                                   /* tp_setattro */
    0,                                   /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                  /* tp_flags */
    "FinalizerClass",                    /* tp_doc */
    0,                                   /* tp_traverse */
    0,                                   /* tp_clear */
    0,                                   /* tp_richcompare */
    0,                                   /* tp_weaklistoffset */
    0,                                   /* tp_iter */
    0,                                   /* tp_iternext */
    0,                                   /* tp_methods */
    0,                                   /* tp_members */
    0,                                   /* tp_getset */
    &PyType_Type,                        /* tp_base */
    0,                                   /* tp_dict */
    0,                                   /* tp_descr_get */
    0,                                   /* tp_descr_set */
    0,                                   /* tp_dictoffset */
    0,                                   /* tp_init */
    0,                                   /* tp_alloc */
    0,                                   /* tp_new */
};

static PyMappingMethods t_fp_as_mapping = {
    (lenfunc)t_fp_map_length,            /* mp_length          */
    (binaryfunc)t_fp_map_get,            /* mp_subscript       */
    (objobjargproc)t_fp_map_set,         /* mp_ass_subscript   */
};

static PySequenceMethods t_fp_as_sequence = {
    (lenfunc)t_fp_seq_length,                 /* sq_length */
    (binaryfunc)t_fp_seq_concat,              /* sq_concat */
    (ssizeargfunc)t_fp_seq_repeat,            /* sq_repeat */
    (ssizeargfunc)t_fp_seq_get,               /* sq_item */
    (ssizessizeargfunc)t_fp_seq_getslice,     /* sq_slice */
    (ssizeobjargproc)t_fp_seq_set,            /* sq_ass_item */
    (ssizessizeobjargproc)t_fp_seq_setslice,  /* sq_ass_slice */
    (objobjproc)t_fp_seq_contains,            /* sq_contains */
    (binaryfunc)t_fp_seq_inplace_concat,      /* sq_inplace_concat */
    (ssizeargfunc)t_fp_seq_inplace_repeat,    /* sq_inplace_repeat */
};

PyTypeObject PY_TYPE(FinalizerProxy) = {
    PyObject_HEAD_INIT(NULL)
    0,                                   /* ob_size */
    "jcc.FinalizerProxy",                /* tp_name */
    sizeof(t_fp),                        /* tp_basicsize */
    0,                                   /* tp_itemsize */
    (destructor)t_fp_dealloc,            /* tp_dealloc */
    0,                                   /* tp_print */
    0,                                   /* tp_getattr */
    0,                                   /* tp_setattr */
    0,                                   /* tp_compare */
    (reprfunc)t_fp_repr,                 /* tp_repr */
    0,                                   /* tp_as_number */
    &t_fp_as_sequence,                   /* tp_as_sequence */
    &t_fp_as_mapping,                    /* tp_as_mapping */
    0,                                   /* tp_hash  */
    0,                                   /* tp_call */
    0,                                   /* tp_str */
    (getattrofunc)t_fp_getattro,         /* tp_getattro */
    (setattrofunc)t_fp_setattro,         /* tp_setattro */
    0,                                   /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT |
     Py_TPFLAGS_HAVE_GC),                /* tp_flags */
    "FinalizerProxy",                    /* tp_doc */
    (traverseproc)t_fp_traverse,         /* tp_traverse */
    (inquiry)t_fp_clear,                 /* tp_clear */
    0,                                   /* tp_richcompare */
    0,                                   /* tp_weaklistoffset */
    (getiterfunc)t_fp_iter,              /* tp_iter */
    0,                                   /* tp_iternext */
    0,                                   /* tp_methods */
    0,                                   /* tp_members */
    0,                                   /* tp_getset */
    0,                                   /* tp_base */
    0,                                   /* tp_dict */
    0,                                   /* tp_descr_get */
    0,                                   /* tp_descr_set */
    0,                                   /* tp_dictoffset */
    0,                                   /* tp_init */
    0,                                   /* tp_alloc */
    0,                                   /* tp_new */
};

static PyObject *t_fc_call(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *obj = PyType_Type.tp_call(self, args, kwds);

    if (obj)
    {
        t_fp *fp = (t_fp *) PY_TYPE(FinalizerProxy).tp_alloc(&PY_TYPE(FinalizerProxy), 0);

        fp->object = obj;      /* released by t_fp_clear() */
        obj = (PyObject *) fp;
    }

    return obj;
}

static void t_fp_dealloc(t_fp *self)
{
    if (self->object)
        ((t_JObject *) self->object)->object.weaken$();

    t_fp_clear(self);
    self->ob_type->tp_free((PyObject *) self);
}

static int t_fp_traverse(t_fp *self, visitproc visit, void *arg)
{
    Py_VISIT(self->object);
    return 0;
}

static int t_fp_clear(t_fp *self)
{
    Py_CLEAR(self->object);
    return 0;
}

static PyObject *t_fp_repr(t_fp *self)
{
    return PyObject_Repr(self->object);
}

static PyObject *t_fp_iter(t_fp *self)
{
    return PyObject_GetIter(self->object);
}

static PyObject *t_fp_getattro(t_fp *self, PyObject *name)
{
    return PyObject_GetAttr(self->object, name);
}

static int t_fp_setattro(t_fp *self, PyObject *name, PyObject *value)
{
    return PyObject_SetAttr(self->object, name, value);
}

static Py_ssize_t t_fp_map_length(t_fp *self)
{
    return PyMapping_Size(self->object);
}

static PyObject *t_fp_map_get(t_fp *self, PyObject *key)
{
    return PyObject_GetItem(self->object, key);
}

static int t_fp_map_set(t_fp *self, PyObject *key, PyObject *value)
{
    if (value == NULL)
        return PyObject_DelItem(self->object, key);

    return PyObject_SetItem(self->object, key, value);
}

static Py_ssize_t t_fp_seq_length(t_fp *self)
{
    return PySequence_Length(self->object);
}

static PyObject *t_fp_seq_get(t_fp *self, Py_ssize_t n)
{
    return PySequence_GetItem(self->object, n);
}

static int t_fp_seq_contains(t_fp *self, PyObject *value)
{
    return PySequence_Contains(self->object, value);
}

static PyObject *t_fp_seq_concat(t_fp *self, PyObject *arg)
{
    return PySequence_Concat(self->object, arg);
}

static PyObject *t_fp_seq_repeat(t_fp *self, Py_ssize_t n)
{
    return PySequence_Repeat(self->object, n);
}

static PyObject *t_fp_seq_getslice(t_fp *self, Py_ssize_t low, Py_ssize_t high)
{
    return PySequence_GetSlice(self->object, low, high);
}

static int t_fp_seq_set(t_fp *self, Py_ssize_t i, PyObject *value)
{
    return PySequence_SetItem(self->object, i, value);
}

static int t_fp_seq_setslice(t_fp *self, Py_ssize_t low,
                             Py_ssize_t high, PyObject *arg)
{
    return PySequence_SetSlice(self->object, low, high, arg);
}

static PyObject *t_fp_seq_inplace_concat(t_fp *self, PyObject *arg)
{
    return PySequence_InPlaceConcat(self->object, arg);
}

static PyObject *t_fp_seq_inplace_repeat(t_fp *self, Py_ssize_t n)
{
    return PySequence_InPlaceRepeat(self->object, n);
}


/* const variable descriptor */

class t_descriptor {
public:
    PyObject_HEAD
    int flags;
    union {
        PyObject *value;
        getclassfn initializeClass;
    } access;
};
    
#define DESCRIPTOR_VALUE   0x0001
#define DESCRIPTOR_CLASS   0x0002
#define DESCRIPTOR_GETFN   0x0004
#define DESCRIPTOR_GENERIC 0x0008

static void t_descriptor_dealloc(t_descriptor *self);
static PyObject *t_descriptor___get__(t_descriptor *self,
                                      PyObject *obj, PyObject *type);

static PyMethodDef t_descriptor_methods[] = {
    { NULL, NULL, 0, NULL }
};


PyTypeObject PY_TYPE(ConstVariableDescriptor) = {
    PyObject_HEAD_INIT(NULL)
    0,                                   /* ob_size */
    "jcc.ConstVariableDescriptor",       /* tp_name */
    sizeof(t_descriptor),                /* tp_basicsize */
    0,                                   /* tp_itemsize */
    (destructor)t_descriptor_dealloc,    /* tp_dealloc */
    0,                                   /* tp_print */
    0,                                   /* tp_getattr */
    0,                                   /* tp_setattr */
    0,                                   /* tp_compare */
    0,                                   /* tp_repr */
    0,                                   /* tp_as_number */
    0,                                   /* tp_as_sequence */
    0,                                   /* tp_as_mapping */
    0,                                   /* tp_hash  */
    0,                                   /* tp_call */
    0,                                   /* tp_str */
    0,                                   /* tp_getattro */
    0,                                   /* tp_setattro */
    0,                                   /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                  /* tp_flags */
    "const variable descriptor",         /* tp_doc */
    0,                                   /* tp_traverse */
    0,                                   /* tp_clear */
    0,                                   /* tp_richcompare */
    0,                                   /* tp_weaklistoffset */
    0,                                   /* tp_iter */
    0,                                   /* tp_iternext */
    t_descriptor_methods,                /* tp_methods */
    0,                                   /* tp_members */
    0,                                   /* tp_getset */
    0,                                   /* tp_base */
    0,                                   /* tp_dict */
    (descrgetfunc)t_descriptor___get__,  /* tp_descr_get */
    0,                                   /* tp_descr_set */
    0,                                   /* tp_dictoffset */
    0,                                   /* tp_init */
    0,                                   /* tp_alloc */
    0,                                   /* tp_new */
};

static void t_descriptor_dealloc(t_descriptor *self)
{
    if (self->flags & DESCRIPTOR_VALUE)
    {
        Py_DECREF(self->access.value);
    }
    self->ob_type->tp_free((PyObject *) self);
}

PyObject *make_descriptor(PyTypeObject *value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        Py_INCREF(value);
        self->access.value = (PyObject *) value;
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(getclassfn initializeClass)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        self->access.initializeClass = initializeClass;
        self->flags = DESCRIPTOR_CLASS;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(getclassfn initializeClass, int generics)
{
    t_descriptor *self = (t_descriptor *) make_descriptor(initializeClass);

    if (self && generics)
        self->flags |= DESCRIPTOR_GENERIC;

    return (PyObject *) self;
}

PyObject *make_descriptor(PyObject *value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        self->access.value = value;
        self->flags = DESCRIPTOR_VALUE;
    }
    else
        Py_DECREF(value);

    return (PyObject *) self;
}

PyObject *make_descriptor(PyObject *(*wrapfn)(const jobject &))
{
    return make_descriptor(PyCObject_FromVoidPtr((void *) wrapfn, NULL));
}

PyObject *make_descriptor(boxfn fn)
{
    return make_descriptor(PyCObject_FromVoidPtr((void *) fn, NULL));
}

PyObject *make_descriptor(jboolean b)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        PyObject *value = b ? Py_True : Py_False;
        self->access.value = (PyObject *) value; Py_INCREF(value);
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(jbyte value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        self->access.value = PyInt_FromLong(value);
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(jchar value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        Py_UNICODE pchar = (Py_UNICODE) value;

        self->access.value = PyUnicode_FromUnicode(&pchar, 1);
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(jdouble value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        self->access.value = PyFloat_FromDouble(value);
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(jfloat value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        self->access.value = PyFloat_FromDouble((double) value);
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(jint value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        self->access.value = PyInt_FromLong(value);
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(jlong value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        self->access.value = PyLong_FromLongLong((long long) value);
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

PyObject *make_descriptor(jshort value)
{
    t_descriptor *self = (t_descriptor *)
        PY_TYPE(ConstVariableDescriptor).tp_alloc(&PY_TYPE(ConstVariableDescriptor), 0);

    if (self)
    {
        self->access.value = PyInt_FromLong((short) value);
        self->flags = DESCRIPTOR_VALUE;
    }

    return (PyObject *) self;
}

static PyObject *t_descriptor___get__(t_descriptor *self,
                                      PyObject *obj, PyObject *type)
{
    if (self->flags & DESCRIPTOR_VALUE)
    {
        Py_INCREF(self->access.value);
        return self->access.value;
    }

    if (self->flags & DESCRIPTOR_CLASS)
    {
#ifdef _java_generics
        if (self->flags & DESCRIPTOR_GENERIC)
            return t_Class::wrap_Object(Class(env->getClass(self->access.initializeClass)), (PyTypeObject *) type);
        else
#endif
            return t_Class::wrap_Object(Class(env->getClass(self->access.initializeClass)));
    }

    Py_RETURN_NONE;
}

