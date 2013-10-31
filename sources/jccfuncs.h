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

#ifndef _jccfuncs_H
#define _jccfuncs_H

#ifdef PYTHON

PyObject *__initialize__(PyObject *module, PyObject *args, PyObject *kwds);
PyObject *initVM(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *getVMEnv(PyObject *self);
PyObject *_set_exception_types(PyObject *self, PyObject *args);
PyObject *_set_function_self(PyObject *self, PyObject *args);
PyObject *findClass(PyObject *self, PyObject *args);
PyObject *makeInterface(PyObject *self, PyObject *args);
PyObject *makeClass(PyObject *self, PyObject *args);
PyObject *JArray_Type(PyObject *self, PyObject *arg);

PyMethodDef jcc_funcs[] = {
    { "initVM", (PyCFunction) __initialize__,
      METH_VARARGS | METH_KEYWORDS, NULL },
    { "getVMEnv", (PyCFunction) getVMEnv,
      METH_NOARGS, NULL },
    { "findClass", (PyCFunction) findClass,
      METH_VARARGS, NULL },
    { "makeInterface", (PyCFunction) makeInterface,
      METH_VARARGS, NULL },
    { "makeClass", (PyCFunction) makeClass,
      METH_VARARGS, NULL },
    { "_set_exception_types", (PyCFunction) _set_exception_types,
      METH_VARARGS, NULL },
    { "_set_function_self", (PyCFunction) _set_function_self,
      METH_VARARGS, NULL },
    { "JArray", (PyCFunction) JArray_Type,
      METH_O, NULL },
    { NULL, NULL, 0, NULL }
};

#endif

#endif /* _jccfuncs_H */
