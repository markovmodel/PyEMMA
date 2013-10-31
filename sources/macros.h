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

#ifndef _macros_H
#define _macros_H

#define OBJ_CALL(action)                                                \
    {                                                                   \
        try {                                                           \
            PythonThreadState state(1);                                 \
            action;                                                     \
        } catch (int e) {                                               \
            switch (e) {                                                \
              case _EXC_PYTHON:                                         \
                return NULL;                                            \
              case _EXC_JAVA:                                           \
                return PyErr_SetJavaError();                            \
              default:                                                  \
                throw;                                                  \
            }                                                           \
        }                                                               \
    }

#define INT_CALL(action)                                                \
    {                                                                   \
        try {                                                           \
            PythonThreadState state(1);                                 \
            action;                                                     \
        } catch (int e) {                                               \
            switch (e) {                                                \
              case _EXC_PYTHON:                                         \
                return -1;                                              \
              case _EXC_JAVA:                                           \
                PyErr_SetJavaError();                                   \
                return -1;                                              \
              default:                                                  \
                throw;                                                  \
            }                                                           \
        }                                                               \
    }


#define DECLARE_METHOD(type, name, flags)               \
    { #name, (PyCFunction) type##_##name, flags, "" }

#define DECLARE_GET_FIELD(type, name)           \
    { #name, (getter) type##_get__##name, NULL, "", NULL }

#define DECLARE_SET_FIELD(type, name)           \
    { #name, NULL, (setter) type##_set__##name, "", NULL }

#define DECLARE_GETSET_FIELD(type, name)        \
    { #name, (getter) type##_get__##name, (setter) type##_set__##name, "", NULL }

#define PY_TYPE(name) name##$$Type

#define DECLARE_TYPE(name, t_name, base, javaClass,                         \
                     init, iter, iternext, getset, mapping, sequence)       \
PyTypeObject PY_TYPE(name) = {                                              \
    PyObject_HEAD_INIT(NULL)                                                \
    /* ob_size            */   0,                                           \
    /* tp_name            */   #name,                                       \
    /* tp_basicsize       */   sizeof(t_name),                              \
    /* tp_itemsize        */   0,                                           \
    /* tp_dealloc         */   0,                                           \
    /* tp_print           */   0,                                           \
    /* tp_getattr         */   0,                                           \
    /* tp_setattr         */   0,                                           \
    /* tp_compare         */   0,                                           \
    /* tp_repr            */   0,                                           \
    /* tp_as_number       */   0,                                           \
    /* tp_as_sequence     */   sequence,                                    \
    /* tp_as_mapping      */   mapping,                                     \
    /* tp_hash            */   0,                                           \
    /* tp_call            */   0,                                           \
    /* tp_str             */   0,                                           \
    /* tp_getattro        */   0,                                           \
    /* tp_setattro        */   0,                                           \
    /* tp_as_buffer       */   0,                                           \
    /* tp_flags           */   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    \
    /* tp_doc             */   #t_name" objects",                           \
    /* tp_traverse        */   0,                                           \
    /* tp_clear           */   0,                                           \
    /* tp_richcompare     */   0,                                           \
    /* tp_weaklistoffset  */   0,                                           \
    /* tp_iter            */   (getiterfunc) iter,                          \
    /* tp_iternext        */   (iternextfunc) iternext,                     \
    /* tp_methods         */   t_name##__methods_,                          \
    /* tp_members         */   0,                                           \
    /* tp_getset          */   getset,                                      \
    /* tp_base            */   &PY_TYPE(base),                              \
    /* tp_dict            */   0,                                           \
    /* tp_descr_get       */   0,                                           \
    /* tp_descr_set       */   0,                                           \
    /* tp_dictoffset      */   0,                                           \
    /* tp_init            */   (initproc)init,                              \
    /* tp_alloc           */   0,                                           \
    /* tp_new             */   0,                                           \
};                                                                          \
PyObject *t_name::wrap_Object(const javaClass& object)                  \
{                                                                       \
    if (!!object)                                                       \
    {                                                                   \
        t_name *self =                                                  \
            (t_name *) PY_TYPE(name).tp_alloc(&PY_TYPE(name), 0);       \
        if (self)                                                       \
            self->object = object;                                      \
        return (PyObject *) self;                                       \
    }                                                                   \
    Py_RETURN_NONE;                                                     \
}                                                                       \
PyObject *t_name::wrap_jobject(const jobject& object)                   \
{                                                                       \
    if (!!object)                                                       \
    {                                                                   \
        if (!env->isInstanceOf(object, javaClass::initializeClass))     \
        {                                                               \
            PyErr_SetObject(PyExc_TypeError,                            \
                            (PyObject *) &PY_TYPE(name));               \
            return NULL;                                                \
        }                                                               \
        t_name *self = (t_name *)                                       \
            PY_TYPE(name).tp_alloc(&PY_TYPE(name), 0);                  \
        if (self)                                                       \
            self->object = javaClass(object);                           \
        return (PyObject *) self;                                       \
    }                                                                   \
    Py_RETURN_NONE;                                                     \
}                                                                       \


#define INSTALL_TYPE(name, module)                                      \
    if (PyType_Ready(&PY_TYPE(name)) == 0)                              \
    {                                                                   \
        Py_INCREF(&PY_TYPE(name));                                      \
        PyModule_AddObject(module, #name, (PyObject *) &PY_TYPE(name)); \
    }


#define Py_RETURN_BOOL(b)                       \
    {                                           \
        if (b)                                  \
            Py_RETURN_TRUE;                     \
        else                                    \
            Py_RETURN_FALSE;                    \
    }

#define Py_RETURN_SELF                                      \
    {                                                       \
        Py_INCREF(self);                                    \
        return (PyObject *) self;                           \
    }


#if PY_VERSION_HEX < 0x02040000

#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#define Py_RETURN_TRUE return Py_INCREF(Py_True), Py_True
#define Py_RETURN_FALSE return Py_INCREF(Py_False), Py_False

#define Py_CLEAR(op)                            \
    do {                                        \
        if (op) {                               \
            PyObject *tmp = (PyObject *)(op);   \
            (op) = NULL;                        \
            Py_DECREF(tmp);                     \
        }                                       \
    } while (0)

#define Py_VISIT(op)                                    \
    do {                                                \
        if (op) {                                       \
            int vret = visit((PyObject *)(op), arg);    \
            if (vret)                                   \
                return vret;                            \
        }                                               \
    } while (0)
          
#endif /* Python 2.3.5 */


#endif /* _macros_H */
