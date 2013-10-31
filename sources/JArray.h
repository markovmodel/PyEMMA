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

#ifndef _JArray_H
#define _JArray_H

#ifdef PYTHON
#include <Python.h>
#include "macros.h"

#if PY_VERSION_HEX < 0x02050000
typedef int Py_ssize_t;
#endif

extern jobjectArray fromPySequence(jclass cls, PyObject *sequence);
extern PyObject *PyErr_SetJavaError();

extern PyTypeObject *PY_TYPE(JArrayObject);
extern PyTypeObject *PY_TYPE(JArrayString);
extern PyTypeObject *PY_TYPE(JArrayBool);
extern PyTypeObject *PY_TYPE(JArrayByte);
extern PyTypeObject *PY_TYPE(JArrayChar);
extern PyTypeObject *PY_TYPE(JArrayDouble);
extern PyTypeObject *PY_TYPE(JArrayFloat);
extern PyTypeObject *PY_TYPE(JArrayInt);
extern PyTypeObject *PY_TYPE(JArrayLong);
extern PyTypeObject *PY_TYPE(JArrayShort);

#else

typedef int Py_ssize_t;

#endif /* PYTHON */

#include "JCCEnv.h"
#include "java/lang/Object.h"


template<typename T> class JArray : public java::lang::Object {
public:
    Py_ssize_t length;

    explicit JArray<T>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jobjectArray) this$) : 0;
    }
    JArray<T>(const JArray<T>& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

#ifdef PYTHON
    JArray<T>(PyObject *sequence) : java::lang::Object(fromPySequence(env->getClass(T::initializeClass), sequence)) {
        length = this$ ? env->getArrayLength((jobjectArray) this$) : 0;
    }

    JArray<T>(jclass cls, PyObject *sequence) : java::lang::Object(fromPySequence(cls, sequence)) {
        length = this$ ? env->getArrayLength((jobjectArray) this$) : 0;
    }

    PyObject *toSequence(PyObject *(*wrapfn)(const T&))
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        PyObject *list = PyList_New(length);

        for (Py_ssize_t i = 0; i < length; i++)
            PyList_SET_ITEM(list, i, (*wrapfn)((*this)[i]));

        return list;
    }

    PyObject *get(Py_ssize_t n, PyObject *(*wrapfn)(const T&))
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
                return (*wrapfn)((*this)[n]);
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }
#endif

    T operator[](Py_ssize_t n) {
        return T(env->getObjectArrayElement((jobjectArray) this$, n));
    }
};

template<> class JArray<jobject> : public java::lang::Object {
  public:
    Py_ssize_t length;

    JArray<jobject>(jclass cls, Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewObjectArray(n, cls, NULL)) {
        length = env->getArrayLength((jobjectArray) this$);
    }

    JArray<jobject>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jobjectArray) this$) : 0;
    }

    JArray<jobject>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

#ifdef PYTHON
    JArray<jobject>(jclass cls, PyObject *sequence) : java::lang::Object(fromPySequence(cls, sequence)) {
        length = this$ ? env->getArrayLength((jobjectArray) this$) : 0;
    }

    PyObject *toSequence(PyObject *(*wrapfn)(const jobject&))
    {
        return toSequence(0, length, wrapfn);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi,
                         PyObject *(*wrapfn)(const jobject&))
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        PyObject *list = PyList_New(hi - lo);

        if (!wrapfn)
            wrapfn = java::lang::t_Object::wrap_jobject;

        for (Py_ssize_t i = lo; i < hi; i++) {
            jobject jobj = env->getObjectArrayElement((jobjectArray) this$, i);
            PyObject *obj = (*wrapfn)(jobj);

            PyList_SET_ITEM(list, i - lo, obj);
        }
         
        return list;
    }

    PyObject *get(Py_ssize_t n, PyObject *(*wrapfn)(const jobject&))
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                if (!wrapfn)
                    wrapfn = java::lang::t_Object::wrap_jobject;

                jobject jobj =
                    env->getObjectArrayElement((jobjectArray) this$, n);

                return (*wrapfn)(jobj);
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                jobject jobj;

                if (PyString_Check(obj) || PyUnicode_Check(obj))
                    jobj = env->fromPyString(obj);
                else if (!PyObject_TypeCheck(obj, &PY_TYPE(JObject)))
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    return -1;
                }
                else
                    jobj = ((t_JObject *) obj)->object.this$;

                try {
                    env->setObjectArrayElement((jobjectArray) this$, n, jobj);
                } catch (int e) {
                    switch (e) {
                      case _EXC_JAVA:
                        PyErr_SetJavaError();
                        return -1;
                      default:
                        throw;
                    }
                }

                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap(PyObject *(*wrapfn)(const jobject&)) const;
#endif

    jobject operator[](Py_ssize_t n) {
        return (jobject) env->getObjectArrayElement((jobjectArray) this$, n);
    }
};

template<> class JArray<jstring> : public java::lang::Object {
  public:
    Py_ssize_t length;

    JArray<jstring>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jobjectArray) this$) : 0;
    }

    JArray<jstring>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jstring>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewObjectArray(n, env->findClass("java/lang/String"), NULL)) {
        length = env->getArrayLength((jobjectArray) this$);
    }

#ifdef PYTHON
    JArray<jstring>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewObjectArray(PySequence_Length(sequence), env->findClass("java/lang/String"), NULL)) {
        length = env->getArrayLength((jobjectArray) this$);

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject *obj = PySequence_GetItem(sequence, i);

            if (obj == NULL)
                break;

            jstring str = env->fromPyString(obj);

            Py_DECREF(obj);
            if (PyErr_Occurred())
                break;

            env->setObjectArrayElement((jobjectArray) this$, i, str);
            env->get_vm_env()->DeleteLocalRef(str);
        }
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        PyObject *list = PyList_New(hi - lo);

        for (Py_ssize_t i = lo; i < hi; i++) {
            jstring str = (jstring)
                env->getObjectArrayElement((jobjectArray) this$, i);
            PyObject *obj = env->fromJString(str, 1);

            PyList_SET_ITEM(list, i - lo, obj);
        }
         
        return list;
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                jstring str = (jstring)
                    env->getObjectArrayElement((jobjectArray) this$, n);
                PyObject *obj = env->fromJString(str, 1);

                return obj;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                jstring str = env->fromPyString(obj);

                if (PyErr_Occurred())
                    return -1;

                env->setObjectArrayElement((jobjectArray) this$, n, str);
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jstring operator[](Py_ssize_t n) {
        return (jstring) env->getObjectArrayElement((jobjectArray) this$, n);
    }
};

template<> class JArray<jboolean> : public java::lang::Object {
  public:
    Py_ssize_t length;

    class arrayElements {
    private:
        jboolean isCopy;
        jbooleanArray array;
        jboolean *elts;
    public:
        arrayElements(jbooleanArray array) {
            this->array = array;
            elts = env->get_vm_env()->GetBooleanArrayElements(array, &isCopy);
        }
        virtual ~arrayElements() {
            env->get_vm_env()->ReleaseBooleanArrayElements(array, elts, 0);
        }
        operator jboolean *() {
            return elts;
        }
    };

    arrayElements elements() {
        return arrayElements((jbooleanArray) this$);
    }

    JArray<jboolean>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jarray) this$) : 0;
    }

    JArray<jboolean>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jboolean>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewBooleanArray(n)) {
        length = env->getArrayLength((jarray) this$);
    }

#ifdef PYTHON
    JArray<jboolean>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewBooleanArray(PySequence_Length(sequence))) {
        length = env->getArrayLength((jarray) this$);
        arrayElements elts = elements();
        jboolean *buf = (jboolean *) elts;

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject *obj = PySequence_GetItem(sequence, i);

            if (!obj)
                break;

            if (obj == Py_True || obj == Py_False)
            {
                buf[i] = (jboolean) (obj == Py_True);
                Py_DECREF(obj);
            }
            else
            {
                PyErr_SetObject(PyExc_TypeError, obj);
                Py_DECREF(obj);
                break;
            }
        }
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        PyObject *list = PyList_New(hi - lo);
        arrayElements elts = elements();
        jboolean *buf = (jboolean *) elts;

        for (Py_ssize_t i = lo; i < hi; i++) {
            jboolean value = buf[i];
            PyObject *obj = value ? Py_True : Py_False;

            Py_INCREF(obj);
            PyList_SET_ITEM(list, i - lo, obj);
        }
         
        return list;
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
                Py_RETURN_BOOL(elements()[n]);
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                elements()[n] = (jboolean) PyObject_IsTrue(obj);
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jboolean operator[](Py_ssize_t n) {
        JNIEnv *vm_env = env->get_vm_env();
        jboolean isCopy = 0;
        jboolean *elts = (jboolean *)
            vm_env->GetPrimitiveArrayCritical((jarray) this$, &isCopy);
        jboolean value = elts[n];

        vm_env->ReleasePrimitiveArrayCritical((jarray) this$, elts, 0);

        return value;
    }
};

template<> class JArray<jbyte> : public java::lang::Object {
  public:
    Py_ssize_t length;

    class arrayElements {
    private:
        jboolean isCopy;
        jbyteArray array;
        jbyte *elts;
    public:
        arrayElements(jbyteArray array) {
            this->array = array;
            elts = env->get_vm_env()->GetByteArrayElements(array, &isCopy);
        }
        virtual ~arrayElements() {
            env->get_vm_env()->ReleaseByteArrayElements(array, elts, 0);
        }
        operator jbyte *() {
            return elts;
        }
    };

    arrayElements elements() {
        return arrayElements((jbyteArray) this$);
    }

    JArray<jbyte>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jarray) this$) : 0;
    }

    JArray<jbyte>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jbyte>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewByteArray(n)) {
        length = env->getArrayLength((jarray) this$);
    }

#ifdef PYTHON
    JArray<jbyte>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewByteArray(PySequence_Length(sequence))) {
        length = env->getArrayLength((jarray) this$);
        arrayElements elts = elements();
        jbyte *buf = (jbyte *) elts;

        if (PyString_Check(sequence))
            memcpy(buf, PyString_AS_STRING(sequence), length);
        else
            for (Py_ssize_t i = 0; i < length; i++) {
                PyObject *obj = PySequence_GetItem(sequence, i);

                if (!obj)
                    break;

                if (PyString_Check(obj) && (PyString_GET_SIZE(obj) == 1))
                {
                    buf[i] = (jbyte) PyString_AS_STRING(obj)[0];
                    Py_DECREF(obj);
                }
                else if (PyInt_CheckExact(obj))
                {
                    buf[i] = (jbyte) PyInt_AS_LONG(obj);
                    Py_DECREF(obj);
                }
                else
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    Py_DECREF(obj);
                    break;
                }
            }
    }

    char getType()
    {
        return 'Z';
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        arrayElements elts = elements();
        jbyte *buf = (jbyte *) elts;
        PyObject *tuple = PyTuple_New(hi - lo);
        
        for (Py_ssize_t i = 0; i < hi - lo; i++)
            PyTuple_SET_ITEM(tuple, i, PyInt_FromLong(buf[lo + i]));

        return tuple;
    }

    PyObject *to_string_()
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        arrayElements elts = elements();
        jbyte *buf = (jbyte *) elts;

        return PyString_FromStringAndSize((char *) buf, length);
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                jbyte b = (*this)[n];
                return PyInt_FromLong(b);
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                if (!PyInt_CheckExact(obj))
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    return -1;
                }

                elements()[n] = (jbyte) PyInt_AS_LONG(obj);
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jbyte operator[](Py_ssize_t n) {
        JNIEnv *vm_env = env->get_vm_env();
        jboolean isCopy = 0;
        jbyte *elts = (jbyte *)
            vm_env->GetPrimitiveArrayCritical((jarray) this$, &isCopy);
        jbyte value = elts[n];

        vm_env->ReleasePrimitiveArrayCritical((jarray) this$, elts, 0);

        return value;
    }
};

template<> class JArray<jchar> : public java::lang::Object {
  public:
    Py_ssize_t length;

    class arrayElements {
    private:
        jboolean isCopy;
        jcharArray array;
        jchar *elts;
    public:
        arrayElements(jcharArray array) {
            this->array = array;
            elts = env->get_vm_env()->GetCharArrayElements(array, &isCopy);
        }
        virtual ~arrayElements() {
            env->get_vm_env()->ReleaseCharArrayElements(array, elts, 0);
        }
        operator jchar *() {
            return elts;
        }
    };

    arrayElements elements() {
        return arrayElements((jcharArray) this$);
    }

    JArray<jchar>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jarray) this$) : 0;
    }

    JArray<jchar>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jchar>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewCharArray(n)) {
        length = env->getArrayLength((jarray) this$);
    }

#ifdef PYTHON
    JArray<jchar>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewCharArray(PySequence_Length(sequence))) {
        length = env->getArrayLength((jarray) this$);
        arrayElements elts = elements();
        jchar *buf = (jchar *) elts;

        if (PyUnicode_Check(sequence))
        {
            if (sizeof(Py_UNICODE) == sizeof(jchar))
                memcpy(buf, PyUnicode_AS_UNICODE(sequence),
                       length * sizeof(jchar));
            else
            {
                Py_UNICODE *pchars = PyUnicode_AS_UNICODE(sequence);
                for (Py_ssize_t i = 0; i < length; i++)
                    buf[i] = (jchar) pchars[i];
            }
        }
        else
            for (Py_ssize_t i = 0; i < length; i++) {
                PyObject *obj = PySequence_GetItem(sequence, i);

                if (!obj)
                    break;

                if (PyUnicode_Check(obj) && (PyUnicode_GET_SIZE(obj) == 1))
                {
                    buf[i] = (jchar) PyUnicode_AS_UNICODE(obj)[0];
                    Py_DECREF(obj);
                }
                else
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    Py_DECREF(obj);
                    break;
                }
            }
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        arrayElements elts = elements();
        jchar *buf = (jchar *) elts;

        if (sizeof(Py_UNICODE) == sizeof(jchar))
            return PyUnicode_FromUnicode((const Py_UNICODE *) buf + lo,
                                         hi - lo);
        else
        {
            PyObject *string = PyUnicode_FromUnicode(NULL, hi - lo);
            Py_UNICODE *pchars = PyUnicode_AS_UNICODE(string);

            for (Py_ssize_t i = lo; i < hi; i++)
                pchars[i - lo] = (Py_UNICODE) buf[i];

            return string;
        }
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                jchar c = (*this)[n];

                if (sizeof(Py_UNICODE) == sizeof(jchar))
                    return PyUnicode_FromUnicode((const Py_UNICODE *) &c, 1);
                else
                {
                    PyObject *string = PyUnicode_FromUnicode(NULL, 1);
                    Py_UNICODE *pchars = PyUnicode_AS_UNICODE(string);

                    pchars[0] = (Py_UNICODE) c;

                    return string;
                }
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                if (!PyUnicode_Check(obj))
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    return -1;
                }
                if (PyUnicode_GET_SIZE(obj) != 1)
                {
                    PyErr_SetObject(PyExc_ValueError, obj);
                    return -1;
                }

                elements()[n] = (jchar) PyUnicode_AS_UNICODE(obj)[0];
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jchar operator[](Py_ssize_t n) {
        JNIEnv *vm_env = env->get_vm_env();
        jboolean isCopy = 0;
        jchar *elts = (jchar *)
            vm_env->GetPrimitiveArrayCritical((jarray) this$, &isCopy);
        jchar value = elts[n];

        vm_env->ReleasePrimitiveArrayCritical((jarray) this$, elts, 0);

        return value;
    }
};

template<> class JArray<jdouble> : public java::lang::Object {
  public:
    Py_ssize_t length;

    class arrayElements {
    private:
        jboolean isCopy;
        jdoubleArray array;
        jdouble *elts;
    public:
        arrayElements(jdoubleArray array) {
            this->array = array;
            elts = env->get_vm_env()->GetDoubleArrayElements(array, &isCopy);
        }
        virtual ~arrayElements() {
            env->get_vm_env()->ReleaseDoubleArrayElements(array, elts, 0);
        }
        operator jdouble *() {
            return elts;
        }
    };

    arrayElements elements() {
        return arrayElements((jdoubleArray) this$);
    }

    JArray<jdouble>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jarray) this$) : 0;
    }

    JArray<jdouble>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jdouble>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewDoubleArray(n)) {
        length = env->getArrayLength((jarray) this$);
    }

#ifdef PYTHON
    JArray<jdouble>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewDoubleArray(PySequence_Length(sequence))) {
        length = env->getArrayLength((jarray) this$);
        arrayElements elts = elements();
        jdouble *buf = (jdouble *) elts;

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject *obj = PySequence_GetItem(sequence, i);

            if (!obj)
                break;

            if (PyFloat_Check(obj))
            {
                buf[i] = (jdouble) PyFloat_AS_DOUBLE(obj);
                Py_DECREF(obj);
            }
            else
            {
                PyErr_SetObject(PyExc_TypeError, obj);
                Py_DECREF(obj);
                break;
            }
        }
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        PyObject *list = PyList_New(hi - lo);
        arrayElements elts = elements();
        jdouble *buf = (jdouble *) elts;

        for (Py_ssize_t i = lo; i < hi; i++)
            PyList_SET_ITEM(list, i - lo, PyFloat_FromDouble((double) buf[i]));

        return list;
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
                return PyFloat_FromDouble((double) (*this)[n]);
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                if (!PyFloat_Check(obj))
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    return -1;
                }

                elements()[n] = (jdouble) PyFloat_AS_DOUBLE(obj);
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jdouble operator[](Py_ssize_t n) {
        JNIEnv *vm_env = env->get_vm_env();
        jboolean isCopy = 0;
        jdouble *elts = (jdouble *)
            vm_env->GetPrimitiveArrayCritical((jarray) this$, &isCopy);
        jdouble value = elts[n];

        vm_env->ReleasePrimitiveArrayCritical((jarray) this$, elts, 0);

        return value;
    }
};

template<> class JArray<jfloat> : public java::lang::Object {
  public:
    Py_ssize_t length;

    class arrayElements {
    private:
        jboolean isCopy;
        jfloatArray array;
        jfloat *elts;
    public:
        arrayElements(jfloatArray array) {
            this->array = array;
            elts = env->get_vm_env()->GetFloatArrayElements(array, &isCopy);
        }
        virtual ~arrayElements() {
            env->get_vm_env()->ReleaseFloatArrayElements(array, elts, 0);
        }
        operator jfloat *() {
            return elts;
        }
    };

    arrayElements elements() {
        return arrayElements((jfloatArray) this$);
    }

    JArray<jfloat>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jarray) this$) : 0;
    }

    JArray<jfloat>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jfloat>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewFloatArray(n)) {
        length = env->getArrayLength((jarray) this$);
    }

#ifdef PYTHON
    JArray<jfloat>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewFloatArray(PySequence_Length(sequence))) {
        length = env->getArrayLength((jarray) this$);
        arrayElements elts = elements();
        jfloat *buf = (jfloat *) elts;

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject *obj = PySequence_GetItem(sequence, i);

            if (!obj)
                break;

            if (PyFloat_Check(obj))
            {
                buf[i] = (jfloat) PyFloat_AS_DOUBLE(obj);
                Py_DECREF(obj);
            }
            else
            {
                PyErr_SetObject(PyExc_TypeError, obj);
                Py_DECREF(obj);
                break;
            }
        }
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        PyObject *list = PyList_New(hi - lo);
        arrayElements elts = elements();
        jfloat *buf = (jfloat *) elts;

        for (Py_ssize_t i = lo; i < hi; i++)
            PyList_SET_ITEM(list, i - lo, PyFloat_FromDouble((double) buf[i]));

        return list;
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
                return PyFloat_FromDouble((double) (*this)[n]);
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                if (!PyFloat_Check(obj))
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    return -1;
                }

                elements()[n] = (jfloat) PyFloat_AS_DOUBLE(obj);
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jfloat operator[](Py_ssize_t n) {
        JNIEnv *vm_env = env->get_vm_env();
        jboolean isCopy = 0;
        jfloat *elts = (jfloat *)
            vm_env->GetPrimitiveArrayCritical((jarray) this$, &isCopy);
        jfloat value = elts[n];

        vm_env->ReleasePrimitiveArrayCritical((jarray) this$, elts, 0);

        return value;
    }
};

template<> class JArray<jint> : public java::lang::Object {
  public:
    Py_ssize_t length;

    class arrayElements {
    private:
        jboolean isCopy;
        jintArray array;
        jint *elts;
    public:
        arrayElements(jintArray array) {
            this->array = array;
            elts = env->get_vm_env()->GetIntArrayElements(array, &isCopy);
        }
        virtual ~arrayElements() {
            env->get_vm_env()->ReleaseIntArrayElements(array, elts, 0);
        }
        operator jint *() {
            return elts;
        }
    };

    arrayElements elements() {
        return arrayElements((jintArray) this$);
    }

    JArray<jint>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jarray) this$) : 0;
    }

    JArray<jint>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jint>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewIntArray(n)) {
        length = env->getArrayLength((jarray) this$);
    }

#ifdef PYTHON
    JArray<jint>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewIntArray(PySequence_Length(sequence))) {
        length = env->getArrayLength((jarray) this$);
        arrayElements elts = elements();
        jint *buf = (jint *) elts;

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject *obj = PySequence_GetItem(sequence, i);

            if (!obj)
                break;

            if (PyInt_Check(obj))
            {
                buf[i] = (jint) PyInt_AS_LONG(obj);
                Py_DECREF(obj);
            }
            else
            {
                PyErr_SetObject(PyExc_TypeError, obj);
                Py_DECREF(obj);
                break;
            }
        }
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        PyObject *list = PyList_New(hi - lo);
        arrayElements elts = elements();
        jint *buf = (jint *) elts;

        for (Py_ssize_t i = lo; i < hi; i++)
            PyList_SET_ITEM(list, i - lo, PyInt_FromLong(buf[i]));

        return list;
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
                return PyInt_FromLong((*this)[n]);
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                if (!PyInt_Check(obj))
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    return -1;
                }

                elements()[n] = (jint) PyInt_AS_LONG(obj);
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jint operator[](Py_ssize_t n) {
        JNIEnv *vm_env = env->get_vm_env();
        jboolean isCopy = 0;
        jint *elts = (jint *)
            vm_env->GetPrimitiveArrayCritical((jarray) this$, &isCopy);
        jint value = elts[n];

        vm_env->ReleasePrimitiveArrayCritical((jarray) this$, elts, 0);

        return value;
    }
};

template<> class JArray<jlong> : public java::lang::Object {
  public:
    Py_ssize_t length;

    class arrayElements {
    private:
        jboolean isCopy;
        jlongArray array;
        jlong *elts;
    public:
        arrayElements(jlongArray array) {
            this->array = array;
            elts = env->get_vm_env()->GetLongArrayElements(array, &isCopy);
        }
        virtual ~arrayElements() {
            env->get_vm_env()->ReleaseLongArrayElements(array, elts, 0);
        }
        operator jlong *() {
            return elts;
        }
    };

    arrayElements elements() {
        return arrayElements((jlongArray) this$);
    }

    JArray<jlong>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jarray) this$) : 0;
    }

    JArray<jlong>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jlong>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewLongArray(n)) {
        length = env->getArrayLength((jarray) this$);
    }

#ifdef PYTHON
    JArray<jlong>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewLongArray(PySequence_Length(sequence))) {
        length = env->getArrayLength((jarray) this$);
        arrayElements elts = elements();
        jlong *buf = (jlong *) elts;

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject *obj = PySequence_GetItem(sequence, i);

            if (!obj)
                break;

            if (PyLong_Check(obj))
            {
                buf[i] = (jlong) PyLong_AsLongLong(obj);
                Py_DECREF(obj);
            }
            else
            {
                PyErr_SetObject(PyExc_TypeError, obj);
                Py_DECREF(obj);
                break;
            }
        }
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        PyObject *list = PyList_New(hi - lo);
        arrayElements elts = elements();
        jlong *buf = (jlong *) elts;

        for (Py_ssize_t i = lo; i < hi; i++)
            PyList_SET_ITEM(list, i - lo, PyLong_FromLongLong((long long) buf[i]));

        return list;
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
                return PyLong_FromLongLong((long long) (*this)[n]);
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                if (!PyLong_Check(obj))
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    return -1;
                }

                elements()[n] = (jlong) PyLong_AsLongLong(obj);
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jlong operator[](long n) {
        JNIEnv *vm_env = env->get_vm_env();
        jboolean isCopy = 0;
        jlong *elts = (jlong *)
            vm_env->GetPrimitiveArrayCritical((jarray) this$, &isCopy);
        jlong value = elts[n];

        vm_env->ReleasePrimitiveArrayCritical((jarray) this$, elts, 0);

        return value;
    }
};

template<> class JArray<jshort> : public java::lang::Object {
  public:
    Py_ssize_t length;

    class arrayElements {
    private:
        jboolean isCopy;
        jshortArray array;
        jshort *elts;
    public:
        arrayElements(jshortArray array) {
            this->array = array;
            elts = env->get_vm_env()->GetShortArrayElements(array, &isCopy);
        }
        virtual ~arrayElements() {
            env->get_vm_env()->ReleaseShortArrayElements(array, elts, 0);
        }
        operator jshort *() {
            return elts;
        }
    };

    arrayElements elements() {
        return arrayElements((jshortArray) this$);
    }

    JArray<jshort>(jobject obj) : java::lang::Object(obj) {
        length = this$ ? env->getArrayLength((jarray) this$) : 0;
    }

    JArray<jshort>(const JArray& obj) : java::lang::Object(obj) {
        length = obj.length;
    }

    JArray<jshort>(Py_ssize_t n) : java::lang::Object(env->get_vm_env()->NewShortArray(n)) {
        length = env->getArrayLength((jarray) this$);
    }

#ifdef PYTHON
    JArray<jshort>(PyObject *sequence) : java::lang::Object(env->get_vm_env()->NewShortArray(PySequence_Length(sequence))) {
        length = env->getArrayLength((jarray) this$);
        arrayElements elts = elements();
        jshort *buf = (jshort *) elts;

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject *obj = PySequence_GetItem(sequence, i);

            if (!obj)
                break;

            if (PyInt_Check(obj))
            {
                buf[i] = (jshort) PyInt_AS_LONG(obj);
                Py_DECREF(obj);
            }
            else
            {
                PyErr_SetObject(PyExc_TypeError, obj);
                Py_DECREF(obj);
                break;
            }
        }
    }

    PyObject *toSequence()
    {
        return toSequence(0, length);
    }

    PyObject *toSequence(Py_ssize_t lo, Py_ssize_t hi)
    {
        if (this$ == NULL)
            Py_RETURN_NONE;

        if (lo < 0) lo = length + lo;
        if (lo < 0) lo = 0;
        else if (lo > length) lo = length;
        if (hi < 0) hi = length + hi;
        if (hi < 0) hi = 0;
        else if (hi > length) hi = length;
        if (lo > hi) lo = hi;

        PyObject *list = PyList_New(hi - lo);
        arrayElements elts = elements();
        jshort *buf = (jshort *) elts;

        for (Py_ssize_t i = lo; i < hi; i++)
            PyList_SET_ITEM(list, i - lo, PyInt_FromLong(buf[i]));

        return list;
    }

    PyObject *get(Py_ssize_t n)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
                return PyInt_FromLong((long) (*this)[n]);
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int set(Py_ssize_t n, PyObject *obj)
    {
        if (this$ != NULL)
        {
            if (n < 0)
                n = length + n;

            if (n >= 0 && n < length)
            {
                if (!PyInt_Check(obj))
                {
                    PyErr_SetObject(PyExc_TypeError, obj);
                    return -1;
                }

                elements()[n] = (jshort) PyInt_AS_LONG(obj);
                return 0;
            }
        }

        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    PyObject *wrap() const;
#endif

    jshort operator[](Py_ssize_t n) {
        JNIEnv *vm_env = env->get_vm_env();
        jboolean isCopy = 0;
        jshort *elts = (jshort *)
            vm_env->GetPrimitiveArrayCritical((jarray) this$, &isCopy);
        jshort value = elts[n];

        vm_env->ReleasePrimitiveArrayCritical((jarray) this$, elts, 0);

        return value;
    }
};

#ifdef PYTHON

template<typename T> class t_JArray {
public:
    PyObject_HEAD
    JArray<T> array;
};

template<typename T, typename U> class t_JArrayWrapper {
public:
    static PyObject *wrap_Object(const JArray<T> &array)
    {
        if (!!array)
            return array.wrap(U::wrap_jobject);

        Py_RETURN_NONE;
    }

    static PyObject *wrap_jobject(const jobject &array)
    {
        if (!!array)
            return JArray<T>(array).wrap(U::wrap_jobject);

        Py_RETURN_NONE;
    }
};

template<typename T> class t_JArrayWrapper<T, T> {
public:
    static PyObject *wrap_Object(const JArray<T> &array)
    {
        if (!!array)
            return array.wrap();

        Py_RETURN_NONE;
    }

    static PyObject *wrap_jobject(const jobject &array)
    {
        if (!!array)
            return JArray<T>(array).wrap();

        Py_RETURN_NONE;
    }
};

#endif

#endif /* _JArray_H */
