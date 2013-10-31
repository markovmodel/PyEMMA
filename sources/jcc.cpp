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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jni.h>

#ifdef linux
#include <dlfcn.h>
#endif

#include <Python.h>
#include "structmember.h"

#include "JObject.h"
#include "JCCEnv.h"
#include "macros.h"

_DLL_EXPORT JCCEnv *env;


/* JCCEnv */

class t_jccenv {
public:
    PyObject_HEAD
    JCCEnv *env;
};
    
static void t_jccenv_dealloc(t_jccenv *self);
static PyObject *t_jccenv_attachCurrentThread(PyObject *self, PyObject *args);
static PyObject *t_jccenv_detachCurrentThread(PyObject *self);
static PyObject *t_jccenv_isCurrentThreadAttached(PyObject *self);
static PyObject *t_jccenv_strhash(PyObject *self, PyObject *arg);
static PyObject *t_jccenv__dumpRefs(PyObject *self,
                                    PyObject *args, PyObject *kwds);
static PyObject *t_jccenv__addClassPath(PyObject *self, PyObject *args);

static PyObject *t_jccenv__get_jni_version(PyObject *self, void *data);
static PyObject *t_jccenv__get_java_version(PyObject *self, void *data);
static PyObject *t_jccenv__get_classpath(PyObject *self, void *data);

static PyGetSetDef t_jccenv_properties[] = {
    { "jni_version", (getter) t_jccenv__get_jni_version, NULL, NULL, NULL },
    { "java_version", (getter) t_jccenv__get_java_version, NULL, NULL, NULL },
    { "classpath", (getter) t_jccenv__get_classpath, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyMemberDef t_jccenv_members[] = {
    { NULL, 0, 0, 0, NULL }
};

static PyMethodDef t_jccenv_methods[] = {
    { "attachCurrentThread", (PyCFunction) t_jccenv_attachCurrentThread,
      METH_VARARGS, NULL },
    { "detachCurrentThread", (PyCFunction) t_jccenv_detachCurrentThread,
      METH_NOARGS, NULL },
    { "isCurrentThreadAttached", (PyCFunction) t_jccenv_isCurrentThreadAttached,
      METH_NOARGS, NULL },
    { "strhash", (PyCFunction) t_jccenv_strhash,
      METH_O, NULL },
    { "_dumpRefs", (PyCFunction) t_jccenv__dumpRefs,
      METH_VARARGS | METH_KEYWORDS, NULL },
    { "_addClassPath", (PyCFunction) t_jccenv__addClassPath,
      METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

PyTypeObject PY_TYPE(JCCEnv) = {
    PyObject_HEAD_INIT(NULL)
    0,                                   /* ob_size */
    "jcc.JCCEnv",                        /* tp_name */
    sizeof(t_jccenv),                    /* tp_basicsize */
    0,                                   /* tp_itemsize */
    (destructor)t_jccenv_dealloc,        /* tp_dealloc */
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
    "JCCEnv",                            /* tp_doc */
    0,                                   /* tp_traverse */
    0,                                   /* tp_clear */
    0,                                   /* tp_richcompare */
    0,                                   /* tp_weaklistoffset */
    0,                                   /* tp_iter */
    0,                                   /* tp_iternext */
    t_jccenv_methods,                    /* tp_methods */
    t_jccenv_members,                    /* tp_members */
    t_jccenv_properties,                 /* tp_getset */
    0,                                   /* tp_base */
    0,                                   /* tp_dict */
    0,                                   /* tp_descr_get */
    0,                                   /* tp_descr_set */
    0,                                   /* tp_dictoffset */
    0,                                   /* tp_init */
    0,                                   /* tp_alloc */
    0,                                   /* tp_new */
};

static void t_jccenv_dealloc(t_jccenv *self)
{
    self->ob_type->tp_free((PyObject *) self);
}

static void add_option(char *name, char *value, JavaVMOption *option)
{
    char *buf = new char[strlen(name) + strlen(value) + 1];

    sprintf(buf, "%s%s", name, value);
    option->optionString = buf;
}

#ifdef _jcc_lib
static void add_paths(char *name, char *p0, char *p1, JavaVMOption *option)
{
#if defined(_MSC_VER) || defined(__WIN32)
    char pathsep = ';';
#else
    char pathsep = ':';
#endif
    char *buf = new char[strlen(name) + strlen(p0) + strlen(p1) + 4];

    sprintf(buf, "%s%s%c%s", name, p0, pathsep, p1);
    option->optionString = buf;
}
#endif


static PyObject *t_jccenv_attachCurrentThread(PyObject *self, PyObject *args)
{
    char *name = NULL;
    int asDaemon = 0, result;

    if (!PyArg_ParseTuple(args, "|si", &name, &asDaemon))
        return NULL;

    result = env->attachCurrentThread(name, asDaemon);
        
    return PyInt_FromLong(result);
}

static PyObject *t_jccenv_detachCurrentThread(PyObject *self)
{
    int result = env->vm->DetachCurrentThread();

    env->set_vm_env(NULL);

    return PyInt_FromLong(result);
}

static PyObject *t_jccenv_isCurrentThreadAttached(PyObject *self)
{
    if (env->get_vm_env() != NULL)
        Py_RETURN_TRUE;

    Py_RETURN_FALSE;
}

static PyObject *t_jccenv_strhash(PyObject *self, PyObject *arg)
{
    int hash = PyObject_Hash(arg);
    char buffer[10];

    sprintf(buffer, "%08x", (unsigned int) hash);
    return PyString_FromStringAndSize(buffer, 8);
}

static PyObject *t_jccenv__dumpRefs(PyObject *self,
                                    PyObject *args, PyObject *kwds)
{
    static char *kwnames[] = {
        "classes", "values", NULL
    };
    int classes = 0, values = 0;
    PyObject *result;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwnames,
                                     &classes, &values))
        return NULL;

    if (classes)
        result = PyDict_New();
    else
        result = PyList_New(env->refs.size());

    int count = 0;

    for (std::multimap<int, countedRef>::iterator iter = env->refs.begin();
         iter != env->refs.end();
         iter++) {
        if (classes)  // return dict of { class name: instance count }
        {
            char *name = env->getClassName(iter->second.global);
            PyObject *key = PyString_FromString(name);
            PyObject *value = PyDict_GetItem(result, key);

            if (value == NULL)
                value = PyInt_FromLong(1);
            else
                value = PyInt_FromLong(PyInt_AS_LONG(value) + 1);

            PyDict_SetItem(result, key, value);
            Py_DECREF(key);
            Py_DECREF(value);

            delete name;
        }
        else if (values)  // return list of (value string, ref count)
        {
            char *str = env->toString(iter->second.global);
            PyObject *key = PyString_FromString(str);
            PyObject *value = PyInt_FromLong(iter->second.count);

#if PY_VERSION_HEX < 0x02040000
            PyList_SET_ITEM(result, count++, Py_BuildValue("(OO)", key, value));
#else
            PyList_SET_ITEM(result, count++, PyTuple_Pack(2, key, value));
#endif
            Py_DECREF(key);
            Py_DECREF(value);

            delete str;
        }
        else  // return list of (id hash code, ref count)
        {
            PyObject *key = PyInt_FromLong(iter->first);
            PyObject *value = PyInt_FromLong(iter->second.count);

#if PY_VERSION_HEX < 0x02040000
            PyList_SET_ITEM(result, count++, Py_BuildValue("(OO)", key, value));
#else
            PyList_SET_ITEM(result, count++, PyTuple_Pack(2, key, value));
#endif
            Py_DECREF(key);
            Py_DECREF(value);
        }
    }

    return result;
}

static PyObject *t_jccenv__addClassPath(PyObject *self, PyObject *args)
{
    const char *classpath;

    if (!PyArg_ParseTuple(args, "s", &classpath))
        return NULL;

    env->setClassPath(classpath);

    Py_RETURN_NONE;
}

static PyObject *t_jccenv__get_jni_version(PyObject *self, void *data)
{
    return PyInt_FromLong(env->getJNIVersion());
}

static PyObject *t_jccenv__get_java_version(PyObject *self, void *data)
{
    return env->fromJString(env->getJavaVersion(), 1);
}

static PyObject *t_jccenv__get_classpath(PyObject *self, void *data)
{
    char *classpath = env->getClassPath();

    if (classpath)
    {
        PyObject *result = PyString_FromString(classpath);

        free(classpath);
        return result;
    }

    Py_RETURN_NONE;
}

_DLL_EXPORT PyObject *getVMEnv(PyObject *self)
{
    if (env->vm != NULL)
    {
        t_jccenv *jccenv = (t_jccenv *) PY_TYPE(JCCEnv).tp_alloc(&PY_TYPE(JCCEnv), 0);
        jccenv->env = env;

        return (PyObject *) jccenv;
    }

    Py_RETURN_NONE;
}

#ifdef _jcc_lib
static void registerNatives(JNIEnv *vm_env);
#endif

_DLL_EXPORT PyObject *initJCC(PyObject *module)
{
    static int _once_only = 1;
#if defined(_MSC_VER) || defined(__WIN32)
#define verstring(n) #n
    PyObject *ver = PyString_FromString(verstring(JCC_VER));
#else
    PyObject *ver = PyString_FromString(JCC_VER);
#endif
    PyObject_SetAttrString(module, "JCC_VERSION", ver); Py_DECREF(ver);

    if (_once_only)
    {
        PyEval_InitThreads();
        INSTALL_TYPE(JCCEnv, module);

        if (env == NULL)
            env = new JCCEnv(NULL, NULL);

        _once_only = 0;
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}

_DLL_EXPORT PyObject *initVM(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwnames[] = {
        "classpath", "initialheap", "maxheap", "maxstack",
        "vmargs", NULL
    };
    char *classpath = NULL;
    char *initialheap = NULL, *maxheap = NULL, *maxstack = NULL;
    PyObject *vmargs = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|zzzzO", kwnames,
                                     &classpath,
                                     &initialheap, &maxheap, &maxstack,
                                     &vmargs))
        return NULL;

    if (env->vm)
    {
        PyObject *module_cp = NULL;

        if (initialheap || maxheap || maxstack || vmargs)
        {
            PyErr_SetString(PyExc_ValueError,
                            "JVM is already running, options are ineffective");
            return NULL;
        }

        if (classpath == NULL && self != NULL)
        {
            module_cp = PyObject_GetAttrString(self, "CLASSPATH");
            if (module_cp != NULL)
                classpath = PyString_AsString(module_cp);
        }

        if (classpath && classpath[0])
            env->setClassPath(classpath);

        Py_XDECREF(module_cp);

        return getVMEnv(self);
    }
    else
    {
        JavaVMInitArgs vm_args;
        JavaVMOption vm_options[32];
        JNIEnv *vm_env;
        JavaVM *vm;
        unsigned int nOptions = 0;
        PyObject *module_cp = NULL;

        vm_args.version = JNI_VERSION_1_4;
        JNI_GetDefaultJavaVMInitArgs(&vm_args);

        if (classpath == NULL && self != NULL)
        {
            module_cp = PyObject_GetAttrString(self, "CLASSPATH");
            if (module_cp != NULL)
                classpath = PyString_AsString(module_cp);
        }

#ifdef _jcc_lib
        PyObject *jcc = PyImport_ImportModule("jcc");
        PyObject *cp = PyObject_GetAttrString(jcc, "CLASSPATH");

        if (classpath)
            add_paths("-Djava.class.path=", PyString_AsString(cp), classpath,
                      &vm_options[nOptions++]);
        else
            add_option("-Djava.class.path=", PyString_AsString(cp),
                       &vm_options[nOptions++]);
            
        Py_DECREF(cp);
        Py_DECREF(jcc);
#else
        if (classpath)
            add_option("-Djava.class.path=", classpath,
                       &vm_options[nOptions++]);
#endif

        Py_XDECREF(module_cp);

        if (initialheap)
            add_option("-Xms", initialheap, &vm_options[nOptions++]);
        if (maxheap)
            add_option("-Xmx", maxheap, &vm_options[nOptions++]);
        if (maxstack)
            add_option("-Xss", maxstack, &vm_options[nOptions++]);

        if (vmargs != NULL && PyString_Check(vmargs))
        {
#ifdef _MSC_VER
            char *buf = _strdup(PyString_AS_STRING(vmargs));
#else
            char *buf = strdup(PyString_AS_STRING(vmargs));
#endif
            char *sep = ",";
            char *option;

            for (option = strtok(buf, sep); option != NULL;
                 option = strtok(NULL, sep)) {
                if (nOptions < sizeof(vm_options) / sizeof(JavaVMOption))
                    add_option("", option, &vm_options[nOptions++]);
                else
                {
                    free(buf);
                    for (unsigned int i = 0; i < nOptions; i++)
                        delete vm_options[i].optionString;
                    PyErr_Format(PyExc_ValueError,
                                 "Too many options (> %d)", nOptions);
                    return NULL;
                }
            }
            free(buf);
        }
        else if (vmargs != NULL && PySequence_Check(vmargs))
        {
            PyObject *fast =
                PySequence_Fast(vmargs, "error converting vmargs to a tuple");

            if (fast == NULL)
                return NULL;

            for (int i = 0; i < PySequence_Fast_GET_SIZE(fast); ++i) {
                PyObject *arg = PySequence_Fast_GET_ITEM(fast, i);

                if (PyString_Check(arg))
                {
                    char *option = PyString_AS_STRING(arg);

                    if (nOptions < sizeof(vm_options) / sizeof(JavaVMOption))
                        add_option("", option, &vm_options[nOptions++]);
                    else
                    {
                        for (unsigned int j = 0; j < nOptions; j++)
                            delete vm_options[j].optionString;
                        PyErr_Format(PyExc_ValueError,
                                     "Too many options (> %d)", nOptions);
                        Py_DECREF(fast);
                        return NULL;
                    }
                }
                else
                {
                    for (unsigned int j = 0; j < nOptions; j++)
                        delete vm_options[j].optionString;
                    PyErr_Format(PyExc_TypeError,
                                 "vmargs arg %d is not a string", i);
                    Py_DECREF(fast);
                    return NULL;
                }
            }

            Py_DECREF(fast);
        }
        else if (vmargs != NULL)
        {
            PyErr_SetString(PyExc_TypeError,
                            "vmargs is not a string or sequence");
            return NULL;
        }

        //vm_options[nOptions++].optionString = "-verbose:gc";
        //vm_options[nOptions++].optionString = "-Xcheck:jni";

        vm_args.nOptions = nOptions;
        vm_args.ignoreUnrecognized = JNI_FALSE;
        vm_args.options = vm_options;

        if (JNI_CreateJavaVM(&vm, (void **) &vm_env, &vm_args) < 0)
        {
            for (unsigned int i = 0; i < nOptions; i++)
                delete vm_options[i].optionString;

            PyErr_Format(PyExc_ValueError,
                         "An error occurred while creating Java VM");
            return NULL;
        }

        env->set_vm(vm, vm_env);

        for (unsigned int i = 0; i < nOptions; i++)
            delete vm_options[i].optionString;

        t_jccenv *jccenv = (t_jccenv *) PY_TYPE(JCCEnv).tp_alloc(&PY_TYPE(JCCEnv), 0);
        jccenv->env = env;

#ifdef _jcc_lib
        registerNatives(vm_env);
#endif

        return (PyObject *) jccenv;
    }
}

/* returns borrowed reference */
_DLL_EXPORT PyObject *getJavaModule(PyObject *module,
                                    const char *parent, const char *name) {
    PyObject *modules = PyImport_GetModuleDict();
    PyObject *parent_module, *full_name;

    if (parent[0] == '\0')
    {
        parent_module = NULL;
        full_name = PyString_FromString(name);
    }
    else if ((parent_module = PyDict_GetItemString(modules, parent)) == NULL)
    {
        PyErr_Format(PyExc_ValueError, "Parent module '%s' not found", parent);
        return NULL;
    }
    else
        full_name = PyString_FromFormat("%s.%s", parent, name);

    PyObject *child_module = PyDict_GetItem(modules, full_name);

    if (child_module == NULL)
    {
        child_module = PyModule_New(PyString_AS_STRING(full_name));
        if (child_module != NULL)
        {
            if (parent_module != NULL)
                PyDict_SetItemString(PyModule_GetDict(parent_module),
                                     name, child_module);
            PyDict_SetItem(modules, full_name, child_module);
            Py_DECREF(child_module);  /* borrow reference */
        }
    }
    Py_DECREF(full_name);

    /* During __install__ pass, __file__ is not yet set on module.
     * During __initialize__ pass, __file__ is passed down to child_module.
     */
    if (child_module != NULL)
    {
        PyObject *__file__ = PyString_FromString("__file__");
        PyObject *file = PyDict_GetItem(PyModule_GetDict(module), __file__);

        if (file != NULL)
            PyDict_SetItem(PyModule_GetDict(child_module), __file__, file);
        Py_DECREF(__file__);
    }

    return child_module;
}

#ifdef _jcc_lib

static void raise_error(JNIEnv *vm_env, const char *message)
{
    jclass cls = vm_env->FindClass("org/apache/jcc/PythonException");
    vm_env->ThrowNew(cls, message);
}

static void _PythonVM_init(JNIEnv *vm_env, jobject self,
                           jstring programName, jobjectArray args)
{
    const char *str = vm_env->GetStringUTFChars(programName, JNI_FALSE);
#ifdef linux
    char buf[32];

    // load python runtime for other .so modules to link (such as _time.so)
    sprintf(buf, "libpython%d.%d.so", PY_MAJOR_VERSION, PY_MINOR_VERSION);
    dlopen(buf, RTLD_NOW | RTLD_GLOBAL);
#endif

    Py_SetProgramName((char *) str);

    PyEval_InitThreads();
    Py_Initialize();

    if (args)
    {
        int argc = vm_env->GetArrayLength(args);
        char **argv = (char **) calloc(argc + 1, sizeof(char *));

        argv[0] = (char *) str;
        for (int i = 0; i < argc; i++) {
            jstring arg = (jstring) vm_env->GetObjectArrayElement(args, i);
            argv[i + 1] = (char *) vm_env->GetStringUTFChars(arg, JNI_FALSE);
        }

        PySys_SetArgv(argc + 1, argv);

        for (int i = 0; i < argc; i++) {
            jstring arg = (jstring) vm_env->GetObjectArrayElement(args, i);
            vm_env->ReleaseStringUTFChars(arg, argv[i + 1]);
        }
        free(argv);
    }
    else
        PySys_SetArgv(1, (char **) &str);

    vm_env->ReleaseStringUTFChars(programName, str);
    PyEval_ReleaseLock();
}

static jobject _PythonVM_instantiate(JNIEnv *vm_env, jobject self,
                                     jstring moduleName, jstring className)
{
    PythonGIL gil(vm_env);

    const char *modStr = vm_env->GetStringUTFChars(moduleName, JNI_FALSE);
    PyObject *module =
        PyImport_ImportModule((char *) modStr);  // python 2.4 cast

    vm_env->ReleaseStringUTFChars(moduleName, modStr);

    if (!module)
    {
        raise_error(vm_env, "import failed");
        return NULL;
    }

    const char *clsStr = vm_env->GetStringUTFChars(className, JNI_FALSE);
    PyObject *cls =
        PyObject_GetAttrString(module, (char *) clsStr); // python 2.4 cast
    PyObject *obj;
    jobject jobj;

    vm_env->ReleaseStringUTFChars(className, clsStr);
    Py_DECREF(module);

    if (!cls)
    {
        raise_error(vm_env, "class not found");
        return NULL;
    }

    obj = PyObject_CallFunctionObjArgs(cls, NULL);
    Py_DECREF(cls);

    if (!obj)
    {
        raise_error(vm_env, "instantiation failed");
        return NULL;
    }

    PyObject *cObj = PyObject_GetAttrString(obj, "_jobject");

    if (!cObj)
    {
        raise_error(vm_env, "instance does not proxy a java object");
        Py_DECREF(obj);

        return NULL;
    }

    jobj = (jobject) PyCObject_AsVoidPtr(cObj);
    Py_DECREF(cObj);

    jobj = vm_env->NewLocalRef(jobj);
    Py_DECREF(obj);

    return jobj;
}

extern "C" {

    JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved)
    {
        JNIEnv *vm_env;

        if (!vm->GetEnv((void **) &vm_env, JNI_VERSION_1_4))
            env = new JCCEnv(vm, vm_env);

        registerNatives(vm_env);

        return JNI_VERSION_1_4;
    }

    JNIEXPORT void JNICALL Java_org_apache_jcc_PythonVM_init(JNIEnv *vm_env, jobject self, jstring programName, jobjectArray args)
    {
        return _PythonVM_init(vm_env, self, programName, args);
    }

    JNIEXPORT jobject JNICALL Java_org_apache_jcc_PythonVM_instantiate(JNIEnv *vm_env, jobject self, jstring moduleName, jstring className)
    {
        return _PythonVM_instantiate(vm_env, self, moduleName, className);
    }

    JNIEXPORT jint JNICALL Java_org_apache_jcc_PythonVM_acquireThreadState(JNIEnv *vm_env)
    {
        PyGILState_STATE state = PyGILState_Ensure();
        PyThreadState *tstate = PyGILState_GetThisThreadState();
        int result = -1;

        if (tstate != NULL && tstate->gilstate_counter >= 1)
            result = ++tstate->gilstate_counter;

        PyGILState_Release(state);
        return result;
    }

    JNIEXPORT jint JNICALL Java_org_apache_jcc_PythonVM_releaseThreadState(JNIEnv *vm_env)
    {
        PyGILState_STATE state = PyGILState_Ensure();
        PyThreadState *tstate = PyGILState_GetThisThreadState();
        int result = -1;

        if (tstate != NULL && tstate->gilstate_counter >= 1)
            result = --tstate->gilstate_counter;

        PyGILState_Release(state);
        return result;
    }
};

static void JNICALL _PythonException_getErrorInfo(JNIEnv *vm_env, jobject self)
{
    PythonGIL gil(vm_env);

    if (!PyErr_Occurred())
        return;

    PyObject *type, *value, *tb, *errorName;
    jclass jcls = vm_env->GetObjectClass(self);

    PyErr_Fetch(&type, &value, &tb);

    errorName = PyObject_GetAttrString(type, "__name__");
    if (errorName != NULL)
    {
        jfieldID fid =
            vm_env->GetFieldID(jcls, "errorName", "Ljava/lang/String;");
        jstring str = env->fromPyString(errorName);

        vm_env->SetObjectField(self, fid, str);
        vm_env->DeleteLocalRef(str);
        Py_DECREF(errorName);
    }

    if (value != NULL)
    {
        PyObject *message = PyObject_Str(value);

        if (message != NULL)
        {
            jfieldID fid =
                vm_env->GetFieldID(jcls, "message", "Ljava/lang/String;");
            jstring str = env->fromPyString(message);

            vm_env->SetObjectField(self, fid, str);
            vm_env->DeleteLocalRef(str);
            Py_DECREF(message);
        }
    }

    PyObject *module = NULL, *cls = NULL, *stringIO = NULL, *result = NULL;
    PyObject *_stderr = PySys_GetObject("stderr");
    if (!_stderr)
        goto err;

    module = PyImport_ImportModule("cStringIO");
    if (!module)
        goto err;

    cls = PyObject_GetAttrString(module, "StringIO");
    Py_DECREF(module);
    if (!cls)
        goto err;

    stringIO = PyObject_CallObject(cls, NULL);
    Py_DECREF(cls);
    if (!stringIO)
        goto err;

    Py_INCREF(_stderr);
    PySys_SetObject("stderr", stringIO);

    PyErr_Restore(type, value, tb);
    PyErr_Print();

    result = PyObject_CallMethod(stringIO, "getvalue", NULL);
    Py_DECREF(stringIO);

    if (result != NULL)
    {
        jfieldID fid =
            vm_env->GetFieldID(jcls, "traceback", "Ljava/lang/String;");
        jstring str = env->fromPyString(result);

        vm_env->SetObjectField(self, fid, str);
        vm_env->DeleteLocalRef(str);
        Py_DECREF(result);
    }

    PySys_SetObject("stderr", _stderr);
    Py_DECREF(_stderr);

    return;

  err:
    PyErr_Restore(type, value, tb);
}

static void JNICALL _PythonException_clear(JNIEnv *vm_env, jobject self)
{
    PythonGIL gil(vm_env);
    PyErr_Clear();
}

static void registerNatives(JNIEnv *vm_env)
{
    jclass cls = vm_env->FindClass("org/apache/jcc/PythonException");
    JNINativeMethod methods[] = {
        { "getErrorInfo", "()V", (void *) _PythonException_getErrorInfo },
        { "clear", "()V", (void *) _PythonException_clear },
    };

    vm_env->RegisterNatives(cls, methods, 2);
}

#endif /* _jcc_lib */
