/*
 *   Copyright (c) 2007-2008 Open Source Applications Foundation
 *
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

#ifndef _JCCEnv_H
#define _JCCEnv_H

#include <stdarg.h>
#if defined(_MSC_VER) || defined(__WIN32)
#define _DLL_IMPORT __declspec(dllimport)
#define _DLL_EXPORT __declspec(dllexport)
#include <windows.h>
#undef MAX_PRIORITY
#undef MIN_PRIORITY
#else
#include <pthread.h>
#define _DLL_IMPORT
#define _DLL_EXPORT
#endif

#ifdef __SUNPRO_CC
#undef DEFAULT_TYPE
#endif

#ifdef TRUE
#undef TRUE
#endif
#ifdef FALSE
#undef FALSE
#endif

#include <map>

#ifdef PYTHON
#include <Python.h>
#endif

#undef EOF

class JCCEnv;

#if defined(_MSC_VER) || defined(__WIN32)

#ifdef _jcc_shared
_DLL_IMPORT extern JCCEnv *env;
_DLL_IMPORT extern DWORD VM_ENV;
#else
_DLL_EXPORT extern JCCEnv *env;
_DLL_EXPORT extern DWORD VM_ENV;
#endif

#else

extern JCCEnv *env;

#endif

#define _EXC_PYTHON ((int) 0)
#define _EXC_JAVA   ((int) 1)

typedef jclass (*getclassfn)(bool);

class countedRef {
public:
    jobject global;
    int count;
};

class _DLL_EXPORT JCCEnv {
protected:
    jclass _sys, _obj, _thr;
    jclass _boo, _byt, _cha, _dou, _flo, _int, _lon, _sho;
    jmethodID *_mids;

    enum {
        mid_sys_identityHashCode,
        mid_sys_setProperty,
        mid_sys_getProperty,
        mid_obj_toString,
        mid_obj_hashCode,
        mid_obj_getClass,
        mid_iterator,
        mid_iterator_next,
        mid_enumeration_nextElement,
        mid_Boolean_booleanValue,
        mid_Byte_byteValue,
        mid_Character_charValue,
        mid_Double_doubleValue,
        mid_Float_floatValue,
        mid_Integer_intValue,
        mid_Long_longValue,
        mid_Short_shortValue,
        mid_Boolean_init,
        mid_Byte_init,
        mid_Character_init,
        mid_Double_init,
        mid_Float_init,
        mid_Integer_init,
        mid_Long_init,
        mid_Short_init,
        max_mid
    };

public:
    JavaVM *vm;
    std::multimap<int, countedRef> refs;
    int handlers;

    explicit JCCEnv(JavaVM *vm, JNIEnv *env);

#if defined(_MSC_VER) || defined(__WIN32)
    inline JNIEnv *get_vm_env() const
    {
        return (JNIEnv *) TlsGetValue(VM_ENV);
    }
#else
    static pthread_key_t VM_ENV;

    inline JNIEnv *get_vm_env() const
    {
        return (JNIEnv *) pthread_getspecific(VM_ENV);
    }
#endif
    void set_vm(JavaVM *vm, JNIEnv *vm_env);
    void set_vm_env(JNIEnv *vm_env);
    int attachCurrentThread(char *name, int asDaemon);

    jint getJNIVersion() const;
    jstring getJavaVersion() const;

    jclass findClass(const char *className) const;
    jboolean isInstanceOf(jobject obj, getclassfn initializeClass) const;

    void registerNatives(jclass cls, JNINativeMethod *methods, int n) const;

    jobject iterator(jobject obj) const;
    jobject iteratorNext(jobject obj) const;
    jobject enumerationNext(jobject obj) const;

    jobject newGlobalRef(jobject obj, int id);
    jobject deleteGlobalRef(jobject obj, int id);

    jclass getClass(getclassfn initializeClass) const;
    jobject newObject(getclassfn initializeClass, jmethodID **mids, int m, ...);

    jobjectArray newObjectArray(jclass cls, int size);
    void setObjectArrayElement(jobjectArray a, int n,
                                       jobject obj) const;
    jobject getObjectArrayElement(jobjectArray a, int n) const;
    int getArrayLength(jarray a) const;

    void reportException() const;

    jobject callObjectMethod(jobject obj, jmethodID mid, ...) const;
    jboolean callBooleanMethod(jobject obj, jmethodID mid, ...) const;
    jbyte callByteMethod(jobject obj, jmethodID mid, ...) const;
    jchar callCharMethod(jobject obj, jmethodID mid, ...) const;
    jdouble callDoubleMethod(jobject obj, jmethodID mid, ...) const;
    jfloat callFloatMethod(jobject obj, jmethodID mid, ...) const;
    jint callIntMethod(jobject obj, jmethodID mid, ...) const;
    jlong callLongMethod(jobject obj, jmethodID mid, ...) const;
    jshort callShortMethod(jobject obj, jmethodID mid, ...) const;
    void callVoidMethod(jobject obj, jmethodID mid, ...) const;

    jobject callNonvirtualObjectMethod(jobject obj, jclass cls,
                                       jmethodID mid, ...) const;
    jboolean callNonvirtualBooleanMethod(jobject obj, jclass cls,
                                         jmethodID mid, ...) const;
    jbyte callNonvirtualByteMethod(jobject obj, jclass cls,
                                   jmethodID mid, ...) const;
    jchar callNonvirtualCharMethod(jobject obj, jclass cls,
                                   jmethodID mid, ...) const;
    jdouble callNonvirtualDoubleMethod(jobject obj, jclass cls,
                                       jmethodID mid, ...) const;
    jfloat callNonvirtualFloatMethod(jobject obj, jclass cls,
                                     jmethodID mid, ...) const;
    jint callNonvirtualIntMethod(jobject obj, jclass cls,
                                 jmethodID mid, ...) const;
    jlong callNonvirtualLongMethod(jobject obj, jclass cls,
                                   jmethodID mid, ...) const;
    jshort callNonvirtualShortMethod(jobject obj, jclass cls,
                                     jmethodID mid, ...) const;
    void callNonvirtualVoidMethod(jobject obj, jclass cls,
                                  jmethodID mid, ...) const;

    jobject callStaticObjectMethod(jclass cls, jmethodID mid, ...) const;
    jboolean callStaticBooleanMethod(jclass cls, jmethodID mid, ...) const;
    jbyte callStaticByteMethod(jclass cls, jmethodID mid, ...) const;
    jchar callStaticCharMethod(jclass cls, jmethodID mid, ...) const;
    jdouble callStaticDoubleMethod(jclass cls, jmethodID mid, ...) const;
    jfloat callStaticFloatMethod(jclass cls, jmethodID mid, ...) const;
    jint callStaticIntMethod(jclass cls, jmethodID mid, ...) const;
    jlong callStaticLongMethod(jclass cls, jmethodID mid, ...) const;
    jshort callStaticShortMethod(jclass cls, jmethodID mid, ...) const;
    void callStaticVoidMethod(jclass cls, jmethodID mid, ...) const;

    jboolean booleanValue(jobject obj) const;
    jbyte byteValue(jobject obj) const;
    jchar charValue(jobject obj) const;
    jdouble doubleValue(jobject obj) const;
    jfloat floatValue(jobject obj) const;
    jint intValue(jobject obj) const;
    jlong longValue(jobject obj) const;
    jshort shortValue(jobject obj) const;

    jobject boxBoolean(jboolean value) const;
    jobject boxByte(jbyte value) const;
    jobject boxChar(jchar value) const;
    jobject boxDouble(jdouble value) const;
    jobject boxFloat(jfloat value) const;
    jobject boxInteger(jint value) const;
    jobject boxLong(jlong value) const;
    jobject boxShort(jshort value) const;

    jmethodID getMethodID(jclass cls, const char *name,
                                  const char *signature) const;
    jfieldID getFieldID(jclass cls, const char *name,
                                const char *signature) const;
    jmethodID getStaticMethodID(jclass cls, const char *name,
                                const char *signature) const;

    jobject getStaticObjectField(jclass cls, const char *name,
                                         const char *signature) const;
    jboolean getStaticBooleanField(jclass cls, const char *name) const;
    jbyte getStaticByteField(jclass cls, const char *name) const;
    jchar getStaticCharField(jclass cls, const char *name) const;
    jdouble getStaticDoubleField(jclass cls, const char *name) const;
    jfloat getStaticFloatField(jclass cls, const char *name) const;
    jint getStaticIntField(jclass cls, const char *name) const;
    jlong getStaticLongField(jclass cls, const char *name) const;
    jshort getStaticShortField(jclass cls, const char *name) const;

    jobject getObjectField(jobject obj, jfieldID id) const;
    jboolean getBooleanField(jobject obj, jfieldID id) const;
    jbyte getByteField(jobject obj, jfieldID id) const;
    jchar getCharField(jobject obj, jfieldID id) const;
    jdouble getDoubleField(jobject obj, jfieldID id) const;
    jfloat getFloatField(jobject obj, jfieldID id) const;
    jint getIntField(jobject obj, jfieldID id) const;
    jlong getLongField(jobject obj, jfieldID id) const;
    jshort getShortField(jobject obj, jfieldID id) const;

    void setObjectField(jobject obj, jfieldID id, jobject value) const;
    void setBooleanField(jobject obj, jfieldID id, jboolean value) const;
    void setByteField(jobject obj, jfieldID id, jbyte value) const;
    void setCharField(jobject obj, jfieldID id, jchar value) const;
    void setDoubleField(jobject obj, jfieldID id, jdouble value) const;
    void setFloatField(jobject obj, jfieldID id, jfloat value) const;
    void setIntField(jobject obj, jfieldID id, jint value) const;
    void setLongField(jobject obj, jfieldID id, jlong value) const;
    void setShortField(jobject obj, jfieldID id, jshort value) const;

    int id(jobject obj) const {
        return obj
            ? get_vm_env()->CallStaticIntMethod(_sys,
                                                _mids[mid_sys_identityHashCode],
                                                obj)
            : 0;
    }

    int hash(jobject obj) const {
        return obj
            ? get_vm_env()->CallIntMethod(obj, _mids[mid_obj_hashCode])
            : 0;
    }

    void setClassPath(const char *classPath);
    char *getClassPath();

    jstring fromUTF(const char *bytes) const;
    char *toUTF(jstring str) const;
    char *toString(jobject obj) const;
    char *getClassName(jobject obj) const;
#ifdef PYTHON
    jclass getPythonExceptionClass() const;
    jstring fromPyString(PyObject *object) const;
    PyObject *fromJString(jstring js, int delete_local_ref) const;
    void finalizeObject(JNIEnv *jenv, PyObject *obj);
#endif

    inline int isSame(jobject o1, jobject o2) const
    {
        return o1 == o2 || get_vm_env()->IsSameObject(o1, o2);
    }
};

#ifdef PYTHON

class PythonGIL {
  private:
    PyGILState_STATE state;
  public:
    PythonGIL()
    {
        state = PyGILState_Ensure();
    }
    PythonGIL(JNIEnv *vm_env)
    {
        state = PyGILState_Ensure();
        env->set_vm_env(vm_env);
    }
    ~PythonGIL()
    {
        PyGILState_Release(state);
    }
};

class PythonThreadState {
  private:
    PyThreadState *state;
    int handler;
  public:
    PythonThreadState(int handler=0)
    {
        state = PyEval_SaveThread();
        this->handler = handler;
        env->handlers += handler;
    }
    ~PythonThreadState()
    {
        PyEval_RestoreThread(state);
        env->handlers -= handler;
    }
};

#endif

#endif /* _JCCEnv_H */
