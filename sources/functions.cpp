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
#include <stdarg.h>
#ifdef _MSC_VER
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif

#include "java/lang/Object.h"
#include "java/lang/Class.h"
#include "java/lang/String.h"
#include "java/lang/Throwable.h"
#include "java/lang/Boolean.h"
#include "java/lang/Byte.h"
#include "java/lang/Character.h"
#include "java/lang/Double.h"
#include "java/lang/Float.h"
#include "java/lang/Integer.h"
#include "java/lang/Long.h"
#include "java/lang/Short.h"
#include "java/util/Iterator.h"
#include "JArray.h"
#include "functions.h"
#include "macros.h"

using namespace java::lang;
using namespace java::util;

PyObject *PyExc_JavaError = PyExc_ValueError;
PyObject *PyExc_InvalidArgsError = PyExc_ValueError;

PyObject *_set_exception_types(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "OO",
                          &PyExc_JavaError, &PyExc_InvalidArgsError))
        return NULL;

    Py_RETURN_NONE;
}

PyObject *_set_function_self(PyObject *self, PyObject *args)
{
    PyObject *object, *module;

    if (!PyArg_ParseTuple(args, "OO", &object, &module))
        return NULL;

    if (!PyCFunction_Check(object))
    {
        PyErr_SetObject(PyExc_TypeError, object);
        return NULL;
    }

    PyCFunctionObject *cfn = (PyCFunctionObject *) object;

    Py_INCREF(module);
    Py_XDECREF(cfn->m_self);
    cfn->m_self = module;

    Py_RETURN_NONE;
}

PyObject *findClass(PyObject *self, PyObject *args)
{
    char *className;

    if (!PyArg_ParseTuple(args, "s", &className))
        return NULL;

    try {
        jclass cls = env->findClass(className);

        if (cls)
            return t_Class::wrap_Object(Class(cls));
    } catch (int e) {
        switch (e) {
          case _EXC_PYTHON:
            return NULL;
          case _EXC_JAVA:
            return PyErr_SetJavaError();
          default:
            throw;
        }
    }

    Py_RETURN_NONE;
}

static const char interface_bytes[] = {
    '\xca', '\xfe', '\xba', '\xbe',   // magic number: 0xcafebabe
    '\x00', '\x00', '\x00', '\x32',   // version 50.0
    '\x00', '\x07',                   // constant pool max index: 6
    '\x07', '\x00', '\x04',           // 1: class name at 4
    '\x07', '\x00', '\x05',           // 2: class name at 5
    '\x07', '\x00', '\x06',           // 3: class name at 6
    '\x01', '\x00', '\x00',           // 4: empty string
    '\x01', '\x00', '\x10',           // 5: 16-byte string: java/lang/Object
    'j', 'a', 'v', 'a', '/', 'l', 'a', 'n', 'g', '/',
    'O', 'b', 'j', 'e', 'c', 't',
    '\x01', '\x00', '\x00',           // 6: empty string
    '\x06', '\x01',                   // public abstract interface
    '\x00', '\x01',                   // this class at 1
    '\x00', '\x02',                   // superclass at 2
    '\x00', '\x01',                   // 1 interface
    '\x00', '\x03',                   // interface at 3
    '\x00', '\x00',                   // 0 fields
    '\x00', '\x00',                   // 0 methods
    '\x00', '\x00'                    // 0 attributes
};

/* make an empty interface that extends an interface */
PyObject *makeInterface(PyObject *self, PyObject *args)
{
    char *name, *extName;
    int name_len, extName_len;

    if (!PyArg_ParseTuple(args, "s#s#",
                          &name, &name_len, &extName, &extName_len))
        return NULL;
    
    JNIEnv *vm_env = env->get_vm_env();
    jclass _ucl = (jclass) vm_env->FindClass("java/net/URLClassLoader");
    jmethodID mid = vm_env->GetStaticMethodID(_ucl, "getSystemClassLoader",
                                              "()Ljava/lang/ClassLoader;");
    jobject classLoader = vm_env->CallStaticObjectMethod(_ucl, mid);
    const int bytes_len = sizeof(interface_bytes);
    const int len = bytes_len + name_len + extName_len;
    char *buf = (char *) malloc(len);

    if (buf == NULL)
        return PyErr_NoMemory();

    int name_pos = 22;
    int extName_pos = 44;
    jclass cls;

    memcpy(buf, interface_bytes, name_pos);
    memcpy(buf + name_pos + name_len, interface_bytes + name_pos,
           extName_pos - name_pos);
    memcpy(buf + extName_pos + name_len + extName_len,
           interface_bytes + extName_pos, bytes_len - extName_pos);
    extName_pos += name_len;

    *((unsigned short *) (buf + name_pos - 2)) = htons(name_len);
    memcpy(buf + name_pos, name, name_len);

    *((unsigned short *) (buf + extName_pos - 2)) = htons(extName_len);
    memcpy(buf + extName_pos, extName, extName_len);

    cls = vm_env->DefineClass(name, classLoader, (const jbyte *) buf, len);
    free(buf);

    if (cls)
        return t_Class::wrap_Object(Class(cls));

    return PyErr_SetJavaError();
}

static const char class_bytes[] = {
    '\xca', '\xfe', '\xba', '\xbe',          // magic number: 0xcafebabe
    '\x00', '\x00', '\x00', '\x32',          // version 50.0
    '\x00', '\x0c',                          // constant pool max index: 11
    '\x0a', '\x00', '\x03', '\x00', '\x08',  // 1: method for class 3 at 8
    '\x07', '\x00', '\x09',                  // 2: class name at 9
    '\x07', '\x00', '\x0a',                  // 3: class name at 10
    '\x07', '\x00', '\x0b',                  // 4: class name at 11
    '\x01', '\x00', '\x06',                  // 5: 6-byte string: <init>
    '<', 'i', 'n', 'i', 't', '>',
    '\x01', '\x00', '\x03', '(', ')', 'V',   // 6: 3-byte string: ()V
    '\x01', '\x00', '\x04',                  // 7: 4-byte string: Code
    'C', 'o', 'd', 'e',
    '\x0c', '\x00', '\x05', '\x00', '\x06',  // 8: name at 5, signature at 6
    '\x01', '\x00', '\x00',                  // 9: empty string
    '\x01', '\x00', '\x00',                  // 10: empty string
    '\x01', '\x00', '\x00',                  // 11: empty string
    '\x00', '\x21',                          // super public
    '\x00', '\x02',                          // this class at 2
    '\x00', '\x03',                          // superclass at 3
    '\x00', '\x01',                          // 1 interface
    '\x00', '\x04',                          // interface at 4
    '\x00', '\x00',                          // 0 fields
    '\x00', '\x01',                          // 1 method
    '\x00', '\x01', '\x00', '\x05',          // public, name at 5
    '\x00', '\x06', '\x00', '\x01',          // signature at 6, 1 attribute
    '\x00', '\x07',                          // attribute name at 7: Code
    '\x00', '\x00', '\x00', '\x11',          // 17 bytes past 6 attribute bytes
    '\x00', '\x01',                          // max stack: 1
    '\x00', '\x01',                          // max locals: 1
    '\x00', '\x00', '\x00', '\x05',          // code length: 5
    '\x2a', '\xb7', '\x00', '\x01', '\xb1',  // actual code bytes
    '\x00', '\x00',                          // 0 method exceptions
    '\x00', '\x00',                          // 0 method attributes
    '\x00', '\x00',                          // 0 attributes
};

/* make an empty class that extends a class and implements an interface */
PyObject *makeClass(PyObject *self, PyObject *args)
{
    char *name, *extName, *implName;
    int name_len, extName_len, implName_len;

    if (!PyArg_ParseTuple(args, "s#s#s#",
                          &name, &name_len, &extName, &extName_len,
                          &implName, &implName_len))
        return NULL;
    
    JNIEnv *vm_env = env->get_vm_env();
    jclass _ucl = (jclass) vm_env->FindClass("java/net/URLClassLoader");
    jmethodID mid = vm_env->GetStaticMethodID(_ucl, "getSystemClassLoader",
                                              "()Ljava/lang/ClassLoader;");
    jobject classLoader = vm_env->CallStaticObjectMethod(_ucl, mid);
    const int bytes_len = sizeof(class_bytes);
    const int len = bytes_len + name_len + extName_len + implName_len;
    char *buf = (char *) malloc(len);

    if (buf == NULL)
        return PyErr_NoMemory();

    int name_pos = 54;
    int extName_pos = 57;
    int implName_pos = 60;
    jclass cls;

    memcpy(buf, class_bytes, name_pos);
    memcpy(buf + name_pos + name_len, class_bytes + name_pos,
           extName_pos - name_pos);
    memcpy(buf + extName_pos + name_len + extName_len,
           class_bytes + extName_pos, bytes_len - extName_pos);
    memcpy(buf + implName_pos + name_len + extName_len + implName_len,
           class_bytes + implName_pos, bytes_len - implName_pos);

    extName_pos += name_len;
    implName_pos += name_len + extName_len;

    *((unsigned short *) (buf + name_pos - 2)) = htons(name_len);
    memcpy(buf + name_pos, name, name_len);

    *((unsigned short *) (buf + extName_pos - 2)) = htons(extName_len);
    memcpy(buf + extName_pos, extName, extName_len);

    *((unsigned short *) (buf + implName_pos - 2)) = htons(implName_len);
    memcpy(buf + implName_pos, implName, implName_len);

    cls = vm_env->DefineClass(name, classLoader, (const jbyte *) buf, len);
    free(buf);

    if (cls)
        return t_Class::wrap_Object(Class(cls));

    return PyErr_SetJavaError();
}

static boxfn get_boxfn(PyTypeObject *type)
{
    static PyObject *boxfn_ = PyString_FromString("boxfn_");
    PyObject *cobj = PyObject_GetAttr((PyObject *) type, boxfn_);
    boxfn fn;
    
    if (cobj == NULL)
        return NULL;

    fn = (boxfn) PyCObject_AsVoidPtr(cobj);
    Py_DECREF(cobj);

    return fn;
}

static int is_instance_of(PyObject *arg, PyTypeObject *type)
{
    static PyObject *class_ = PyString_FromString("class_");
    PyObject *clsObj = PyObject_GetAttr((PyObject *) type, class_);
    int result;

    if (clsObj == NULL)
        return -1;

    result = env->get_vm_env()->
        IsInstanceOf(((t_Object *) arg)->object.this$,
                     (jclass) ((t_Object *) clsObj)->object.this$);
    Py_DECREF(clsObj);

    return result;
}


#if defined(_MSC_VER) || defined(__SUNPRO_CC)
int __parseArgs(PyObject *args, char *types, ...)
{
    int count = ((PyTupleObject *)(args))->ob_size;
    va_list list, check;

    va_start(list, types);
    va_start(check, types);

    return _parseArgs(((PyTupleObject *)(args))->ob_item, count, types,
		      list, check);
}

int __parseArg(PyObject *arg, char *types, ...)
{
    va_list list, check;

    va_start(list, types);
    va_start(check, types);

    return _parseArgs(&arg, 1, types, list, check);
}

int _parseArgs(PyObject **args, unsigned int count, char *types,
	       va_list list, va_list check)
{
    unsigned int typeCount = strlen(types);

    if (count > typeCount)
        return -1;
#else

int _parseArgs(PyObject **args, unsigned int count, char *types, ...)
{
    unsigned int typeCount = strlen(types);
    va_list list, check;

    if (count > typeCount)
        return -1;

    va_start(list, types);
    va_start(check, types);
#endif

    if (!env->vm)
    {
        PyErr_SetString(PyExc_RuntimeError, "initVM() must be called first");
        return -1;
    }

    JNIEnv *vm_env = env->get_vm_env();

    if (!vm_env)
    {
        PyErr_SetString(PyExc_RuntimeError, "attachCurrentThread() must be called first");
        return -1;
    }

    unsigned int pos = 0;
    int array = 0;

    for (unsigned int a = 0; a < count; a++, pos++) {
        PyObject *arg = args[a];
        char tc = types[pos];

        if (array > 1 && tc != '[')
          tc = 'o';

        switch (tc) {
          case '[':
          {
              if (++array > 1 &&
                  !PyObject_TypeCheck(arg, PY_TYPE(JArrayObject)))
                return -1;

              a -= 1;
              break;
          }

          case 'j':           /* Java object, with class$    */
          case 'k':           /* Java object, with initializeClass */
          case 'K':           /* Java object, with initializeClass and params */
          {
              jclass cls = NULL;

              switch (tc) {
                case 'j':
                  cls = (jclass) va_arg(list, Class *)->this$;
                  break;
                case 'k':
                case 'K':
                  try {
                      getclassfn initializeClass = va_arg(list, getclassfn);
                      cls = env->getClass(initializeClass);
                  } catch (int e) {
                      switch (e) {
                        case _EXC_PYTHON:
                          return -1;
                        case _EXC_JAVA:
                          PyErr_SetJavaError();
                          return -1;
                        default:
                          throw;
                      }
                  }
                  break;
              }

              if (arg == Py_None)
                  break;

              /* ensure that class Class is initialized (which may not be the
               * case because of earlier recursion avoidance (JObject(cls)).
               */
              if (!Class::class$)
                  env->getClass(Class::initializeClass);

              if (array)
              {
                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayObject)))
                      break;

                  if (PySequence_Check(arg) &&
                      !PyString_Check(arg) && !PyUnicode_Check(arg))
                  {
                      if (PySequence_Length(arg) > 0)
                      {
                          PyObject *obj = PySequence_GetItem(arg, 0);
                          int ok = 0;

                          if (obj == Py_None)
                              ok = 1;
                          else if (PyObject_TypeCheck(obj, &PY_TYPE(Object)) &&
                                   vm_env->IsInstanceOf(((t_Object *) obj)->object.this$, cls))
                              ok = 1;
                          else if (PyObject_TypeCheck(obj, &PY_TYPE(FinalizerProxy)))
                          {
                              PyObject *o = ((t_fp *) obj)->object;

                              if (PyObject_TypeCheck(o, &PY_TYPE(Object)) &&
                                  vm_env->IsInstanceOf(((t_Object *) o)->object.this$, cls))
                                  ok = 1;
                          }

                          Py_DECREF(obj);
                          if (ok)
                              break;
                      }
                      else
                          break;
                  }
              }
              else if (PyObject_TypeCheck(arg, &PY_TYPE(Object)) &&
                       vm_env->IsInstanceOf(((t_Object *) arg)->object.this$, cls))
                  break;
              else if (PyObject_TypeCheck(arg, &PY_TYPE(FinalizerProxy)))
              {
                  arg = ((t_fp *) arg)->object;
                  if (PyObject_TypeCheck(arg, &PY_TYPE(Object)) &&
                      vm_env->IsInstanceOf(((t_Object *) arg)->object.this$, cls))
                      break;
              }

              return -1;
          }

          case 'Z':           /* boolean, strict */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;

                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayBool)))
                      break;

                  if (PySequence_Check(arg))
                  {
                      if (PySequence_Length(arg) > 0)
                      {
                          PyObject *obj = PySequence_GetItem(arg, 0);
                          int ok = obj == Py_True || obj == Py_False;

                          Py_DECREF(obj);
                          if (ok)
                              break;
                      }
                      else
                          break;
                  }
              }
              else if (arg == Py_True || arg == Py_False)
                  break;

              return -1;
          }

          case 'B':           /* byte */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;
                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayByte)))
                      break;
              }
              else if (PyString_Check(arg) && (PyString_Size(arg) == 1))
                  break;
              else if (PyInt_CheckExact(arg))
                  break;

              return -1;
          }

          case 'C':           /* char */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;
                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayChar)))
                      break;
              }
              else if (PyUnicode_Check(arg) && PyUnicode_GET_SIZE(arg) == 1)
                  break;
              return -1;
          }

          case 'I':           /* int */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;

                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayInt)))
                      break;

                  if (PySequence_Check(arg))
                  {
                      if (PySequence_Length(arg) > 0)
                      {
                          PyObject *obj = PySequence_GetItem(arg, 0);
                          int ok = PyInt_CheckExact(obj);

                          Py_DECREF(obj);
                          if (ok)
                              break;
                      }
                      else
                          break;
                  }
              }
              else if (PyInt_CheckExact(arg))
                  break;

              return -1;
          }

          case 'S':           /* short */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;

                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayShort)))
                      break;

                  if (PySequence_Check(arg))
                  {
                      if (PySequence_Length(arg) > 0)
                      {
                          PyObject *obj = PySequence_GetItem(arg, 0);
                          int ok = PyInt_CheckExact(obj);

                          Py_DECREF(obj);
                          if (ok)
                              break;
                      }
                      else
                          break;
                  }
              }
              else if (PyInt_CheckExact(arg))
                  break;

              return -1;
          }

          case 'D':           /* double */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;

                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayDouble)))
                      break;

                  if (PySequence_Check(arg))
                  {
                      if (PySequence_Length(arg) > 0)
                      {
                          PyObject *obj = PySequence_GetItem(arg, 0);
                          int ok = PyFloat_CheckExact(obj);

                          Py_DECREF(obj);
                          if (ok)
                              break;
                      }
                      else
                          break;
                  }
              }
              else if (PyFloat_CheckExact(arg))
                  break;

              return -1;
          }

          case 'F':           /* float */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;

                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayFloat)))
                      break;

                  if (PySequence_Check(arg))
                  {
                      if (PySequence_Length(arg) > 0)
                      {
                          PyObject *obj = PySequence_GetItem(arg, 0);
                          int ok = PyFloat_CheckExact(obj);

                          Py_DECREF(obj);
                          if (ok)
                              break;
                      }
                      else
                          break;
                  }
              }
              else if (PyFloat_CheckExact(arg))
                  break;

              return -1;
          }

          case 'J':           /* long long */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;

                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayLong)))
                      break;

                  if (PySequence_Check(arg))
                  {
                      if (PySequence_Length(arg) > 0)
                      {
                          PyObject *obj = PySequence_GetItem(arg, 0);
                          int ok = PyLong_CheckExact(obj);

                          Py_DECREF(obj);
                          if (ok)
                              break;
                      }
                      else
                          break;
                  }
              }
              else if (PyLong_CheckExact(arg))
                  break;

              return -1;
          }

          case 's':           /* string  */
          {
              if (array)
              {
                  if (arg == Py_None)
                      break;

                  if (PyObject_TypeCheck(arg, PY_TYPE(JArrayString)))
                      break;

                  if (PySequence_Check(arg) && 
                      !PyString_Check(arg) && !PyUnicode_Check(arg))
                  {
                      if (PySequence_Length(arg) > 0)
                      {
                          PyObject *obj = PySequence_GetItem(arg, 0);
                          int ok =
                              (obj == Py_None ||
                               PyString_Check(obj) || PyUnicode_Check(obj));

                          Py_DECREF(obj);
                          if (ok)
                              break;
                      }
                      else
                          break;
                  }
              }
              else if (arg == Py_None ||
                       PyString_Check(arg) || PyUnicode_Check(arg))
                  break;

              return -1;
          }

          case 'o':         /* java.lang.Object */
            break;

          case 'O':         /* java.lang.Object with type param */
          {
              PyTypeObject *type = va_arg(list, PyTypeObject *);

              if (type != NULL)
              {
                  boxfn fn = get_boxfn(type);
    
                  if (fn == NULL || fn(type, arg, NULL) < 0)
                      return -1;
              }
              break;
          }

          case 'T':         /* tuple of python types with wrapfn_ */
          {
              static PyObject *wrapfn_ = PyString_FromString("wrapfn_");
              int len = va_arg(list, int);

              if (PyTuple_Check(arg))
              {
                  if (PyTuple_GET_SIZE(arg) != len)
                      return -1;

                  for (int i = 0; i < len; i++) {
                      PyObject *type = PyTuple_GET_ITEM(arg, i);

                      if (!(type == Py_None ||
                            (PyType_Check(type) &&
                             PyObject_HasAttr(type, wrapfn_))))
                          return -1;
                  }
                  break;
              }
              return -1;
          }

          default:
            return -1;
        }

        if (tc != '[')
            array = 0;
    }

    if (array)
        return -1;

    if (pos != typeCount)
      return -1;

    pos = 0;

    for (unsigned int a = 0; a < count; a++, pos++) {
        PyObject *arg = args[a];
        char tc = types[pos];

        if (array > 1 && tc != '[')
            tc = 'o';

        switch (tc) {
          case '[':
          {
              array += 1;
              a -= 1;

              break;
          }

          case 'j':           /* Java object except String and Object */
          case 'k':           /* Java object, with initializeClass    */
          case 'K':           /* Java object, with initializeClass and params */
          {
              jclass cls = NULL;

              switch (tc) {
                case 'j':
                  cls = (jclass) va_arg(check, Class *)->this$;
                  break;
                case 'k':
                case 'K':
                  getclassfn initializeClass = va_arg(check, getclassfn);
                  cls = env->getClass(initializeClass);
                  break;
              }

              if (array)
              {
                  JArray<jobject> *array = va_arg(list, JArray<jobject> *);

#ifdef _java_generics
                  if (tc == 'K')
                  {
                      PyTypeObject ***tp = va_arg(list, PyTypeObject ***);

                      va_arg(list, getparametersfn);
                      *tp = NULL;
                  }
#endif
                  if (arg == Py_None)
                      *array = JArray<jobject>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayObject)))
                      *array = ((t_JArray<jobject> *) arg)->array;
                  else 
                      *array = JArray<jobject>(cls, arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  Object *obj = va_arg(list, Object *);

                  if (PyObject_TypeCheck(arg, &PY_TYPE(FinalizerProxy)))
                      arg = ((t_fp *) arg)->object;

#ifdef _java_generics
                  if (tc == 'K')
                  {
                      PyTypeObject ***tp = va_arg(list, PyTypeObject ***);
                      PyTypeObject **(*parameters_)(void *) = 
                          va_arg(list, getparametersfn);

                      if (arg == Py_None)
                          *tp = NULL;
                      else
                          *tp = (*parameters_)(arg);
                  }
#endif

                  *obj = arg == Py_None
                      ? Object(NULL)
                      : ((t_Object *) arg)->object;
              }
              break;
          }

          case 'Z':           /* boolean, strict */
          {
              if (array)
              {
                  JArray<jboolean> *array = va_arg(list, JArray<jboolean> *);

                  if (arg == Py_None)
                      *array = JArray<jboolean>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayBool)))
                      *array = ((t_JArray<jboolean> *) arg)->array;
                  else
                      *array = JArray<jboolean>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  jboolean *b = va_arg(list, jboolean *);
                  *b = arg == Py_True;
              }
              break;
          }

          case 'B':           /* byte */
          {
              if (array)
              {
                  JArray<jbyte> *array = va_arg(list, JArray<jbyte> *);

                  if (arg == Py_None)
                      *array = JArray<jbyte>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayByte)))
                      *array = ((t_JArray<jbyte> *) arg)->array;
                  else 
                      *array = JArray<jbyte>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else if (PyString_Check(arg))
              {
                  jbyte *a = va_arg(list, jbyte *);
                  *a = (jbyte) PyString_AS_STRING(arg)[0];
              }
              else
              {
                  jbyte *a = va_arg(list, jbyte *);
                  *a = (jbyte) PyInt_AsLong(arg);
              }
              break;
          }

          case 'C':           /* char */
          {
              if (array)
              {
                  JArray<jchar> *array = va_arg(list, JArray<jchar> *);

                  if (arg == Py_None)
                      *array = JArray<jchar>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayChar)))
                      *array = ((t_JArray<jchar> *) arg)->array;
                  else 
                      *array = JArray<jchar>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  jchar *c = va_arg(list, jchar *);
                  *c = (jchar) PyUnicode_AS_UNICODE(arg)[0];
              }
              break;
          }

          case 'I':           /* int */
          {
              if (array)
              {
                  JArray<jint> *array = va_arg(list, JArray<jint> *);

                  if (arg == Py_None)
                      *array = JArray<jint>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayInt)))
                      *array = ((t_JArray<jint> *) arg)->array;
                  else 
                      *array = JArray<jint>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  jint *n = va_arg(list, jint *);
                  *n = (jint) PyInt_AsLong(arg);
              }
              break;
          }

          case 'S':           /* short */
          {
              if (array)
              {
                  JArray<jshort> *array = va_arg(list, JArray<jshort> *);

                  if (arg == Py_None)
                      *array = JArray<jshort>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayShort)))
                      *array = ((t_JArray<jshort> *) arg)->array;
                  else 
                      *array = JArray<jshort>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  jshort *n = va_arg(list, jshort *);
                  *n = (jshort) PyInt_AsLong(arg);
              }
              break;
          }

          case 'D':           /* double */
          {
              if (array)
              {
                  JArray<jdouble> *array = va_arg(list, JArray<jdouble> *);

                  if (arg == Py_None)
                      *array = JArray<jdouble>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayDouble)))
                      *array = ((t_JArray<jdouble> *) arg)->array;
                  else 
                      *array = JArray<jdouble>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  jdouble *d = va_arg(list, jdouble *);
                  *d = (jdouble) PyFloat_AsDouble(arg);
              }
              break;
          }

          case 'F':           /* float */
          {
              if (array)
              {
                  JArray<jfloat> *array = va_arg(list, JArray<jfloat> *);

                  if (arg == Py_None)
                      *array = JArray<jfloat>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayFloat)))
                      *array = ((t_JArray<jfloat> *) arg)->array;
                  else 
                      *array = JArray<jfloat>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  jfloat *d = va_arg(list, jfloat *);
                  *d = (jfloat) (float) PyFloat_AsDouble(arg);
              }
              break;
          }

          case 'J':           /* long long */
          {
              if (array)
              {
                  JArray<jlong> *array = va_arg(list, JArray<jlong> *);

                  if (arg == Py_None)
                      *array = JArray<jlong>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayLong)))
                      *array = ((t_JArray<jlong> *) arg)->array;
                  else 
                      *array = JArray<jlong>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  jlong *l = va_arg(list, jlong *);
                  *l = (jlong) PyLong_AsLongLong(arg);
              }
              break;
          }

          case 's':           /* string  */
          {
              if (array)
              {
                  JArray<jstring> *array = va_arg(list, JArray<jstring> *);

                  if (arg == Py_None)
                      *array = JArray<jstring>((jobject) NULL);
                  else if (PyObject_TypeCheck(arg, PY_TYPE(JArrayString)))
                      *array = ((t_JArray<jstring> *) arg)->array;
                  else 
                      *array = JArray<jstring>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  String *str = va_arg(list, String *);

                  if (arg == Py_None)
                      *str = String(NULL);
                  else
                  {
                      *str = p2j(arg);
                      if (PyErr_Occurred())
                          return -1;
                  }
              }
              break;
          }

          case 'o':           /* java.lang.Object  */
          case 'O':           /* java.lang.Object with type param */
          {
              if (array)
              {
                  JArray<Object> *array = va_arg(list, JArray<Object> *);

                  if (arg == Py_None)
                      *array = JArray<Object>((jobject) NULL);
                  else 
                      *array = JArray<Object>(arg);

                  if (PyErr_Occurred())
                      return -1;
              }
              else
              {
                  Object *obj = va_arg(list, Object *);

                  if (tc == 'O')
                  {
                      PyTypeObject *type = va_arg(check, PyTypeObject *);

                      if (type != NULL)
                      {
                          boxfn fn = get_boxfn(type);

                          if (fn == NULL || fn(type, arg, obj) < 0)
                              return -1;

                          break;
                      }
                  }

                  if (boxObject(NULL, arg, obj) < 0)
                      return -1;
              }
              break;
          }

          case 'T':         /* tuple of python types with wrapfn_ */
          {
              int len = va_arg(check, int);
              PyTypeObject **types = va_arg(list, PyTypeObject **);

              for (int i = 0; i < len; i++) {
                  PyObject *type = PyTuple_GET_ITEM(arg, i);

                  if (type == Py_None)
                      types[i] = NULL;
                  else
                      types[i] = (PyTypeObject *) type;
              }
              break;
          }

          default:
            return -1;
        }

        if (tc != '[')
            array = 0;
    }

    if (pos == typeCount)
        return 0;

    return -1;
}


String p2j(PyObject *object)
{
    return String(env->fromPyString(object));
}

PyObject *j2p(const String& js)
{
    return env->fromJString((jstring) js.this$, 0);
}

PyObject *PyErr_SetArgsError(char *name, PyObject *args)
{
    if (!PyErr_Occurred())
    {
        PyObject *err = Py_BuildValue("(sO)", name, args);

        PyErr_SetObject(PyExc_InvalidArgsError, err);
        Py_DECREF(err);
    }

    return NULL;
}

PyObject *PyErr_SetArgsError(PyObject *self, char *name, PyObject *args)
{
    if (!PyErr_Occurred())
    {
        PyObject *type = (PyObject *) self->ob_type;
        PyObject *err = Py_BuildValue("(OsO)", type, name, args);

        PyErr_SetObject(PyExc_InvalidArgsError, err);
        Py_DECREF(err);
    }

    return NULL;
}

PyObject *PyErr_SetArgsError(PyTypeObject *type, char *name, PyObject *args)
{
    if (!PyErr_Occurred())
    {
        PyObject *err = Py_BuildValue("(OsO)", type, name, args);

        PyErr_SetObject(PyExc_InvalidArgsError, err);
        Py_DECREF(err);
    }

    return NULL;
}

PyObject *PyErr_SetJavaError()
{
    JNIEnv *vm_env = env->get_vm_env();
    jthrowable throwable = vm_env->ExceptionOccurred();
    PyObject *err;

    vm_env->ExceptionClear();
    err = t_Throwable::wrap_Object(Throwable(throwable));

    PyErr_SetObject(PyExc_JavaError, err);
    Py_DECREF(err);

    return NULL;
}

void throwPythonError(void)
{
    PyObject *exc = PyErr_Occurred();

    if (exc && PyErr_GivenExceptionMatches(exc, PyExc_JavaError))
    {
        PyObject *value, *traceback;

        PyErr_Fetch(&exc, &value, &traceback);
        if (value)
        {
            PyObject *je = PyObject_CallMethod(value, "getJavaException", "");

            if (!je)
                PyErr_Restore(exc, value, traceback);
            else
            {
                Py_DECREF(exc);
                Py_DECREF(value);
                Py_XDECREF(traceback);
                exc = je;

                if (exc && PyObject_TypeCheck(exc, &PY_TYPE(Throwable)))
                {
                    jobject jobj = ((t_Throwable *) exc)->object.this$;

                    env->get_vm_env()->Throw((jthrowable) jobj);
                    Py_DECREF(exc);

                    return;
                }
            }
        }
        else
        {
            Py_DECREF(exc);
            Py_XDECREF(traceback);
        }
    }
    else if (exc && PyErr_GivenExceptionMatches(exc, PyExc_StopIteration))
    {
        PyErr_Clear();
        return;
    }

    if (exc)
    {
        PyObject *name = PyObject_GetAttrString(exc, "__name__");

        env->get_vm_env()->ThrowNew(env->getPythonExceptionClass(),
                                    PyString_AS_STRING(name));
        Py_DECREF(name);
    }
    else
        env->get_vm_env()->ThrowNew(env->getPythonExceptionClass(),
                                    "python error");
}

void throwTypeError(const char *name, PyObject *object)
{
    PyObject *tuple = Py_BuildValue("(ssO)", "while calling", name, object);

    PyErr_SetObject(PyExc_TypeError, tuple);
    Py_DECREF(tuple);

    env->get_vm_env()->ThrowNew(env->getPythonExceptionClass(), "type error");
}

int abstract_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *err =
        Py_BuildValue("(sO)", "instantiating java class", self->ob_type);

    PyErr_SetObject(PyExc_NotImplementedError, err);
    Py_DECREF(err);

    return -1;
}

PyObject *callSuper(PyTypeObject *type, const char *name, PyObject *args,
                    int cardinality)
{
    PyObject *super = (PyObject *) type->tp_base;
    PyObject *method =
        PyObject_GetAttrString(super, (char *) name); // python 2.4 cast
    PyObject *value;

    if (!method)
        return NULL;

    if (cardinality > 1)
        value = PyObject_Call(method, args, NULL);
    else
    {
#if PY_VERSION_HEX < 0x02040000
        PyObject *tuple = Py_BuildValue("(O)", args);
#else
        PyObject *tuple = PyTuple_Pack(1, args);
#endif   
        value = PyObject_Call(method, tuple, NULL);
        Py_DECREF(tuple);
    }

    Py_DECREF(method);

    return value;
}

PyObject *callSuper(PyTypeObject *type, PyObject *self,
                    const char *name, PyObject *args, int cardinality)
{
#if PY_VERSION_HEX < 0x02040000
    PyObject *tuple = Py_BuildValue("(OO)", type, self);
#else
    PyObject *tuple = PyTuple_Pack(2, type, self);
#endif
    PyObject *super = PyObject_Call((PyObject *) &PySuper_Type, tuple, NULL);
    PyObject *method, *value;

    Py_DECREF(tuple);
    if (!super)
        return NULL;

    method = PyObject_GetAttrString(super, (char *) name); // python 2.4 cast
    Py_DECREF(super);
    if (!method)
        return NULL;

    if (cardinality > 1)
        value = PyObject_Call(method, args, NULL);
    else
    {
#if PY_VERSION_HEX < 0x02040000
        tuple = Py_BuildValue("(O)", args);
#else
        tuple = PyTuple_Pack(1, args);
#endif
        value = PyObject_Call(method, tuple, NULL);
        Py_DECREF(tuple);
    }

    Py_DECREF(method);

    return value;
}

PyObject *castCheck(PyObject *obj, getclassfn initializeClass,
                    int reportError)
{
    if (PyObject_TypeCheck(obj, &PY_TYPE(FinalizerProxy)))
        obj = ((t_fp *) obj)->object;

    if (!PyObject_TypeCheck(obj, &PY_TYPE(Object)))
    {
        if (reportError)
            PyErr_SetObject(PyExc_TypeError, obj);
        return NULL;
    }

    jobject jobj = ((t_Object *) obj)->object.this$;

    if (jobj && !env->isInstanceOf(jobj, initializeClass))
    {
        if (reportError)
            PyErr_SetObject(PyExc_TypeError, obj);

        return NULL;
    }

    return obj;
}

PyObject *get_extension_iterator(PyObject *self)
{
    return PyObject_CallMethod(self, "iterator", "");
}

PyObject *get_extension_next(PyObject *self)
{
    return PyObject_CallMethod(self, "next", "");
}

PyObject *get_extension_nextElement(PyObject *self)
{
    return PyObject_CallMethod(self, "nextElement", "");
}

jobjectArray fromPySequence(jclass cls, PyObject *sequence)
{
    if (sequence == Py_None)
        return NULL;

    if (!PySequence_Check(sequence))
    {
        PyErr_SetObject(PyExc_TypeError, sequence);
        return NULL;
    }

    int length = PySequence_Length(sequence);
    jobjectArray array;

    try {
        array = env->newObjectArray(cls, length);
    } catch (int e) {
        switch (e) {
          case _EXC_PYTHON:
            return NULL;
          case _EXC_JAVA:
            PyErr_SetJavaError();
            return NULL;
          default:
            throw;
        }
    }

    JNIEnv *vm_env = env->get_vm_env();

    for (int i = 0; i < length; i++) {
        PyObject *obj = PySequence_GetItem(sequence, i);
        int deleteLocal = 0;
        jobject jobj;

        if (!obj)
            break;
        else if (obj == Py_None)
            jobj = NULL;
        else if (PyString_Check(obj) || PyUnicode_Check(obj))
        {
            jobj = env->fromPyString(obj);
            deleteLocal = 1;
        }
        else if (PyObject_TypeCheck(obj, &PY_TYPE(JObject)))
            jobj = ((t_JObject *) obj)->object.this$;
        else if (PyObject_TypeCheck(obj, &PY_TYPE(FinalizerProxy)))
            jobj = ((t_JObject *) ((t_fp *) obj)->object)->object.this$;
        else if (obj == Py_True || obj == Py_False)
        {
            jobj = env->boxBoolean(obj == Py_True);
            deleteLocal = 1;
        }
        else if (PyFloat_Check(obj))
        {
            jobj = env->boxDouble(PyFloat_AS_DOUBLE(obj));
            deleteLocal = 1;
        }
        else if (PyInt_Check(obj))
        {
            jobj = env->boxInteger(PyInt_AS_LONG(obj));
            deleteLocal = 1;
        }
        else if (PyLong_Check(obj))
        {
            jobj = env->boxLong(PyLong_AsLongLong(obj));
            deleteLocal = 1;
        }
        else
        {
            PyErr_SetObject(PyExc_TypeError, obj);
            Py_DECREF(obj);
            return NULL;
        }

        try {
            env->setObjectArrayElement(array, i, jobj);
            if (deleteLocal)
                vm_env->DeleteLocalRef(jobj);
            Py_DECREF(obj);
        } catch (int e) {
            Py_DECREF(obj);
            switch (e) {
              case _EXC_JAVA:
                PyErr_SetJavaError();
                return NULL;
              default:
                throw;
            }
        }
    }

    return array;
}

void installType(PyTypeObject *type, PyObject *module, char *name,
                 int isExtension)
{
    if (PyType_Ready(type) == 0)
    {
        Py_INCREF(type);
        if (isExtension)
        {
            type->ob_type = &PY_TYPE(FinalizerClass);
            Py_INCREF(&PY_TYPE(FinalizerClass));
        }
        PyModule_AddObject(module, name, (PyObject *) type);
    }
}

PyObject *wrapType(PyTypeObject *type, const jobject& obj)
{
    static PyObject *wrapfn_ = PyString_FromString("wrapfn_");
    PyObject *cobj = PyObject_GetAttr((PyObject *) type, wrapfn_);
    PyObject *(*wrapfn)(const jobject&);
    
    if (cobj == NULL)
        return NULL;

    wrapfn = (PyObject *(*)(const jobject &)) PyCObject_AsVoidPtr(cobj);
    Py_DECREF(cobj);

    return wrapfn(obj);
}

PyObject *unboxBoolean(const jobject& obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::Boolean::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(Boolean));
            return NULL;
        }
        
        if (env->booleanValue(obj))
            Py_RETURN_TRUE;

        Py_RETURN_FALSE;
    }

    Py_RETURN_NONE;
}

PyObject *unboxByte(const jobject &obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::Byte::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(Byte));
            return NULL;
        }
        
        return PyInt_FromLong((long) env->byteValue(obj));
    }

    Py_RETURN_NONE;
}

PyObject *unboxCharacter(const jobject &obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::Character::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(Character));
            return NULL;
        }
        
        jchar c = env->charValue(obj);
        return PyUnicode_FromUnicode((Py_UNICODE *) &c, 1);
    }

    Py_RETURN_NONE;
}

PyObject *unboxDouble(const jobject &obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::Double::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(Double));
            return NULL;
        }
        
        return PyFloat_FromDouble((double) env->doubleValue(obj));
    }

    Py_RETURN_NONE;
}

PyObject *unboxFloat(const jobject &obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::Float::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(Float));
            return NULL;
        }
        
        return PyFloat_FromDouble((double) env->floatValue(obj));
    }

    Py_RETURN_NONE;
}

PyObject *unboxInteger(const jobject &obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::Integer::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(Integer));
            return NULL;
        }
        
        return PyInt_FromLong((long) env->intValue(obj));
    }

    Py_RETURN_NONE;
}

PyObject *unboxLong(const jobject &obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::Long::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(Long));
            return NULL;
        }
        
        return PyLong_FromLongLong((PY_LONG_LONG) env->longValue(obj));
    }

    Py_RETURN_NONE;
}

PyObject *unboxShort(const jobject &obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::Short::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(Short));
            return NULL;
        }
        
        return PyInt_FromLong((long) env->shortValue(obj));
    }

    Py_RETURN_NONE;
}

PyObject *unboxString(const jobject &obj)
{
    if (obj != NULL)
    {
        if (!env->isInstanceOf(obj, java::lang::String::initializeClass))
        {
            PyErr_SetObject(PyExc_TypeError,
                            (PyObject *) &java::lang::PY_TYPE(String));
            return NULL;
        }
        
        return env->fromJString((jstring) obj, 0);
    }

    Py_RETURN_NONE;
}

static int boxJObject(PyTypeObject *type, PyObject *arg,
                      java::lang::Object *obj)
{
    if (arg == Py_None)
    {
        if (obj != NULL)
            *obj = Object(NULL);
    }
    else if (PyObject_TypeCheck(arg, &PY_TYPE(Object)))
    {
        if (type != NULL && !is_instance_of(arg, type))
            return -1;

        if (obj != NULL)
            *obj = ((t_Object *) arg)->object;
    }
    else if (PyObject_TypeCheck(arg, &PY_TYPE(FinalizerProxy)))
    {
        arg = ((t_fp *) arg)->object;
        if (PyObject_TypeCheck(arg, &PY_TYPE(Object)))
        {
            if (type != NULL && !is_instance_of(arg, type))
                return -1;

            if (obj != NULL)
                *obj = ((t_Object *) arg)->object;
        }
        else
            return -1;
    }
    else
        return 1;

    return 0;
}

int boxBoolean(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (arg == Py_True)
    {
        if (obj != NULL)
            *obj = *Boolean::TRUE;
    }
    else if (arg == Py_False)
    {
        if (obj != NULL)
            *obj = *Boolean::FALSE;
    }
    else
        return -1;

    return 0;
}

int boxByte(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyInt_Check(arg))
    {
        int n = PyInt_AS_LONG(arg);
        jbyte b = (jbyte) n;

        if (b == n)
        {
            if (obj != NULL)
                *obj = Byte(b);
        }
        else
            return -1;
    }
    else if (PyLong_Check(arg))
    {
        PY_LONG_LONG ln = PyLong_AsLongLong(arg);
        jbyte b = (jbyte) ln;

        if (b == ln)
        {
            if (obj != NULL)
                *obj = Byte(b);
        }
        else
            return -1;
    }
    else if (PyFloat_Check(arg))
    {
        double d = PyFloat_AS_DOUBLE(arg);
        jbyte b = (jbyte) d;

        if (b == d)
        {
            if (obj != NULL)
                *obj = Byte(b);
        }
        else
            return -1;
    }
    else
        return -1;

    return 0;
}

int boxCharacter(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyString_Check(arg))
    {
        char *c;
        Py_ssize_t len;

        if (PyString_AsStringAndSize(arg, &c, &len) < 0 || len != 1)
            return -1;

        if (obj != NULL)
            *obj = Character((jchar) c[0]);
    }
    else if (PyUnicode_Check(arg))
    {
        Py_ssize_t len = PyUnicode_GetSize(arg);

        if (len != 1)
            return -1;

        if (obj != NULL)
            *obj = Character((jchar) PyUnicode_AsUnicode(arg)[0]);
    }
    else
        return -1;

    return 0;
}

int boxCharSequence(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyString_Check(arg) || PyUnicode_Check(arg))
    {
        if (obj != NULL)
        {
            *obj = p2j(arg);
            if (PyErr_Occurred())
                return -1;
        }
    }
    else
        return -1;
    
    return 0;
}

int boxDouble(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyInt_Check(arg))
    {
        if (obj != NULL)
            *obj = Double((jdouble) PyInt_AS_LONG(arg));
    }
    else if (PyLong_Check(arg))
    {
        if (obj != NULL)
            *obj = Double((jdouble) PyLong_AsLongLong(arg));
    }
    else if (PyFloat_Check(arg))
    {
        if (obj != NULL)
            *obj = Double(PyFloat_AS_DOUBLE(arg));
    }
    else
        return -1;

    return 0;
}

int boxFloat(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyInt_Check(arg))
    {
        if (obj != NULL)
            *obj = Float((jfloat) PyInt_AS_LONG(arg));
    }
    else if (PyLong_Check(arg))
    {
        PY_LONG_LONG ln = PyLong_AsLongLong(arg);
        float f = (float) ln;

        if ((PY_LONG_LONG) f == ln)
        {
            if (obj != NULL)
                *obj = Float(f);
        }
        else
            return -1;
    }
    else if (PyFloat_Check(arg))
    {
        double d = PyFloat_AS_DOUBLE(arg);
        float f = (float) d;

        if ((double) f == d)
        {
            if (obj != NULL)
                *obj = Float(f);
        }
        else
            return -1;
    }
    else
        return -1;

    return 0;
}

int boxInteger(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyInt_Check(arg))
    {
        if (obj != NULL)
            *obj = Integer((jint) PyInt_AS_LONG(arg));
    }
    else if (PyLong_Check(arg))
    {
        PY_LONG_LONG ln = PyLong_AsLongLong(arg);
        int n = (int) ln;

        if (n == ln)
        {
            if (obj != NULL)
                *obj = Integer(n);
        }
        else
            return -1;
    }
    else if (PyFloat_Check(arg))
    {
        double d = PyFloat_AS_DOUBLE(arg);
        int n = (int) d;

        if ((double) n == d)
        {
            if (obj != NULL)
                *obj = Integer(n);
        }
        else
            return -1;
    }
    else
        return -1;

    return 0;
}

int boxLong(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyInt_Check(arg))
    {
        if (obj != NULL)
            *obj = Long((jlong) PyInt_AS_LONG(arg));
    }
    else if (PyLong_Check(arg))
    {
        if (obj != NULL)
            *obj = Long((jlong) PyLong_AsLongLong(arg));
    }
    else if (PyFloat_Check(arg))
    {
        double d = PyFloat_AS_DOUBLE(arg);
        PY_LONG_LONG n = (PY_LONG_LONG) d;

        if ((double) n == d)
        {
            if (obj != NULL)
                *obj = Long((jlong) n);
        }
        else
            return -1;
    }
    else
        return -1;

    return 0;
}

int boxNumber(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyInt_Check(arg))
    {
        if (obj != NULL)
            *obj = Integer((jint) PyInt_AS_LONG(arg));
    }
    else if (PyLong_Check(arg))
    {
        if (obj != NULL)
            *obj = Long((jlong) PyLong_AsLongLong(arg));
    }
    else if (PyFloat_Check(arg))
    {
        if (obj != NULL)
            *obj = Double((jdouble) PyFloat_AS_DOUBLE(arg));
    }
    else
        return -1;

    return 0;
}

int boxShort(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyInt_Check(arg))
    {
        int n = (int) PyInt_AS_LONG(arg);
        short sn = (short) n;

        if (sn == n)
        {
            if (obj != NULL)
                *obj = Short((jshort) sn);
        }
        else
            return -1;
    }
    else if (PyLong_Check(arg))
    {
        PY_LONG_LONG ln = PyLong_AsLongLong(arg);
        short sn = (short) ln;

        if (sn == ln)
        {
            if (obj != NULL)
                *obj = Short((jshort) sn);
        }
        else
            return -1;
    }
    else if (PyFloat_Check(arg))
    {
        double d = PyFloat_AS_DOUBLE(arg);
        short sn = (short) (int) d;

        if ((double) sn == d)
        {
            if (obj != NULL)
                *obj = Short((jshort) sn);
        }
        else
            return -1;
    }
    else
        return -1;

    return 0;
}

int boxString(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (PyString_Check(arg) || PyUnicode_Check(arg))
    {
        if (obj != NULL)
        {
            *obj = p2j(arg);
            if (PyErr_Occurred())
                return -1;
        }
    }
    else
        return -1;

    return 0;
}

int boxObject(PyTypeObject *type, PyObject *arg, java::lang::Object *obj)
{
    int result = boxJObject(type, arg, obj);

    if (result <= 0)
        return result;

    if (obj != NULL)
    {
        if (PyString_Check(arg) || PyUnicode_Check(arg))
        {
            *obj = p2j(arg);
            if (PyErr_Occurred())
                return -1;
        }
        else if (arg == Py_True)
            *obj = *Boolean::TRUE;
        else if (arg == Py_False)
            *obj = *Boolean::FALSE;
        else if (PyInt_Check(arg))
        {
            long ln = PyInt_AS_LONG(arg);
            int n = (int) ln;

            if (ln != (long) n)
                *obj = Long((jlong) ln);
            else
                *obj = Integer((jint) n);
        }
        else if (PyLong_Check(arg))
            *obj = Long((jlong) PyLong_AsLongLong(arg));
        else if (PyFloat_Check(arg))
            *obj = Double((jdouble) PyFloat_AS_DOUBLE(arg));
        else
            return -1;
    }
    else if (!(PyString_Check(arg) || PyUnicode_Check(arg) ||
               arg == Py_True || arg == Py_False ||
               PyInt_Check(arg) || PyLong_Check(arg) ||
               PyFloat_Check(arg)))
        return -1;

    return 0;
}


#ifdef _java_generics
PyObject *typeParameters(PyTypeObject *types[], size_t size)
{
    size_t count = size / sizeof(PyTypeObject *);
    PyObject *tuple = PyTuple_New(count);

    for (size_t i = 0; i < count; i++) {
        PyObject *type = (PyObject *) types[i];
        
        if (type == NULL)
            type = Py_None;

        PyTuple_SET_ITEM(tuple, i, type);
        Py_INCREF(type);
    }

    return tuple;
}
#endif
