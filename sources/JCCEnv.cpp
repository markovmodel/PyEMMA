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

#include <map>

#include <stdlib.h>
#include <string.h>
#include <jni.h>

#include "JCCEnv.h"

#if defined(_MSC_VER) || defined(__WIN32)
_DLL_EXPORT DWORD VM_ENV = 0;
#else
pthread_key_t JCCEnv::VM_ENV = (pthread_key_t) NULL;
#endif

#if defined(_MSC_VER) || defined(__WIN32)

static CRITICAL_SECTION *mutex = NULL;

class lock {
public:
    lock() {
        EnterCriticalSection(mutex);
    }
    virtual ~lock() {
        LeaveCriticalSection(mutex);
    }
};

#else

static pthread_mutex_t *mutex = NULL;

class lock {
public:
    lock() {
        pthread_mutex_lock(mutex);
    }
    virtual ~lock() {
        pthread_mutex_unlock(mutex);
    }
};

#endif

JCCEnv::JCCEnv(JavaVM *vm, JNIEnv *vm_env)
{
#if defined(_MSC_VER) || defined(__WIN32)
    if (!mutex)
    {
        mutex = new CRITICAL_SECTION();
        InitializeCriticalSection(mutex);  // recursive by default
    }
#else
    if (!mutex)
    {
        pthread_mutexattr_t attr;

        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        mutex = new pthread_mutex_t();
        pthread_mutex_init(mutex, &attr);
    }
#endif

    if (vm)
        set_vm(vm, vm_env);
    else
        this->vm = NULL;
}

void JCCEnv::set_vm(JavaVM *vm, JNIEnv *vm_env)
{
    this->vm = vm;
    set_vm_env(vm_env);

    _sys = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/System"));
    _obj = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Object"));
#ifdef _jcc_lib
    _thr = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("org/apache/jcc/PythonException"));
#else
    _thr = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/RuntimeException"));
#endif

    _boo = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Boolean"));
    _byt = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Byte"));
    _cha = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Character"));
    _dou = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Double"));
    _flo = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Float"));
    _int = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Integer"));
    _lon = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Long"));
    _sho = (jclass) vm_env->NewGlobalRef(vm_env->FindClass("java/lang/Short"));

    _mids = new jmethodID[max_mid];

    _mids[mid_sys_identityHashCode] =
        vm_env->GetStaticMethodID(_sys, "identityHashCode",
                                  "(Ljava/lang/Object;)I");
    _mids[mid_sys_setProperty] =
        vm_env->GetStaticMethodID(_sys, "setProperty",
                                  "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
    _mids[mid_sys_getProperty] =
        vm_env->GetStaticMethodID(_sys, "getProperty",
                                  "(Ljava/lang/String;)Ljava/lang/String;");
    _mids[mid_obj_toString] =
        vm_env->GetMethodID(_obj, "toString",
                            "()Ljava/lang/String;");
    _mids[mid_obj_hashCode] =
        vm_env->GetMethodID(_obj, "hashCode",
                            "()I");
    _mids[mid_obj_getClass] =
        vm_env->GetMethodID(_obj, "getClass",
                            "()Ljava/lang/Class;");

    jclass iterable = vm_env->FindClass("java/lang/Iterable");

    if (iterable == NULL) /* JDK < 1.5 */
    {
        vm_env->ExceptionClear();
        _mids[mid_iterator] = NULL;
        _mids[mid_iterator_next] = NULL;
    }
    else
    {
        _mids[mid_iterator] =
            vm_env->GetMethodID(iterable,
                                "iterator", "()Ljava/util/Iterator;");
        _mids[mid_iterator_next] =
            vm_env->GetMethodID(vm_env->FindClass("java/util/Iterator"),
                                "next", "()Ljava/lang/Object;");
    }

    _mids[mid_enumeration_nextElement] =
        vm_env->GetMethodID(vm_env->FindClass("java/util/Enumeration"),
                            "nextElement", "()Ljava/lang/Object;");

    _mids[mid_Boolean_booleanValue] =
        vm_env->GetMethodID(_boo, "booleanValue", "()Z");
    _mids[mid_Byte_byteValue] = 
        vm_env->GetMethodID(_byt, "byteValue", "()B");
    _mids[mid_Character_charValue] =
        vm_env->GetMethodID(_cha, "charValue", "()C");
    _mids[mid_Double_doubleValue] = 
        vm_env->GetMethodID(_dou, "doubleValue", "()D");
    _mids[mid_Float_floatValue] =
        vm_env->GetMethodID(_flo, "floatValue", "()F");
    _mids[mid_Integer_intValue] = 
        vm_env->GetMethodID(_int, "intValue", "()I");
    _mids[mid_Long_longValue] = 
        vm_env->GetMethodID(_lon, "longValue", "()J");
    _mids[mid_Short_shortValue] = 
        vm_env->GetMethodID(_sho, "shortValue", "()S");

    _mids[mid_Boolean_init] =
        vm_env->GetMethodID(_boo, "<init>", "(Z)V");
    _mids[mid_Byte_init] = 
        vm_env->GetMethodID(_byt, "<init>", "(B)V");
    _mids[mid_Character_init] =
        vm_env->GetMethodID(_cha, "<init>", "(C)V");
    _mids[mid_Double_init] = 
        vm_env->GetMethodID(_dou, "<init>", "(D)V");
    _mids[mid_Float_init] = 
        vm_env->GetMethodID(_flo, "<init>", "(F)V");
    _mids[mid_Integer_init] =
        vm_env->GetMethodID(_int, "<init>", "(I)V");
    _mids[mid_Long_init] = 
        vm_env->GetMethodID(_lon, "<init>", "(J)V");
    _mids[mid_Short_init] = 
        vm_env->GetMethodID(_sho, "<init>", "(S)V");
}

int JCCEnv::attachCurrentThread(char *name, int asDaemon)
{
    JNIEnv *jenv = NULL;
    JavaVMAttachArgs attach = {
        JNI_VERSION_1_4, name, NULL
    };
    int result;

    if (asDaemon)
        result = vm->AttachCurrentThreadAsDaemon((void **) &jenv, &attach);
    else
        result = vm->AttachCurrentThread((void **) &jenv, &attach);

    set_vm_env(jenv);

    return result;
}

#if defined(_MSC_VER) || defined(__WIN32)

void JCCEnv::set_vm_env(JNIEnv *vm_env)
{
    if (!VM_ENV)
        VM_ENV = TlsAlloc();
    TlsSetValue(VM_ENV, (LPVOID) vm_env);
}

#else

void JCCEnv::set_vm_env(JNIEnv *vm_env)
{
    if (!VM_ENV)
        pthread_key_create(&VM_ENV, NULL);
    pthread_setspecific(VM_ENV, (void *) vm_env);
}

#endif

jint JCCEnv::getJNIVersion() const
{
    return get_vm_env()->GetVersion();
}

jstring JCCEnv::getJavaVersion() const
{
    return (jstring)
        callStaticObjectMethod(_sys, _mids[mid_sys_getProperty],
                               get_vm_env()->NewStringUTF("java.version"));
}

jobject JCCEnv::iterator(jobject obj) const
{
    return callObjectMethod(obj, _mids[mid_iterator]);
}

jobject JCCEnv::iteratorNext(jobject obj) const
{
    return callObjectMethod(obj, _mids[mid_iterator_next]);
}

jobject JCCEnv::enumerationNext(jobject obj) const
{
    return callObjectMethod(obj, _mids[mid_enumeration_nextElement]);
}

jboolean JCCEnv::isInstanceOf(jobject obj, getclassfn initializeClass) const
{
    return get_vm_env()->IsInstanceOf(obj, getClass(initializeClass));
}

jclass JCCEnv::findClass(const char *className) const
{
    jclass cls = NULL;

    if (vm)
    {
        JNIEnv *vm_env = get_vm_env();

        if (vm_env)
        {
            cls = vm_env->FindClass(className);
            if (cls == NULL)
                reportException();
        }
#ifdef PYTHON
        else
        {
            PythonGIL gil;

            PyErr_SetString(PyExc_RuntimeError, "attachCurrentThread() must be called first");
            throw _EXC_PYTHON;
        }
#else
        else
            throw _EXC_JAVA;
#endif
    }
#ifdef PYTHON
    else
    {
        PythonGIL gil;

        PyErr_SetString(PyExc_RuntimeError, "initVM() must be called first");
        throw _EXC_PYTHON;
    }
#else
    else
        throw _EXC_JAVA;
#endif

    reportException();

    return cls;
}

void JCCEnv::registerNatives(jclass cls, JNINativeMethod *methods, int n) const
{
    get_vm_env()->RegisterNatives(cls, methods, n);
}

jobject JCCEnv::newGlobalRef(jobject obj, int id)
{
    if (obj)
    {
        if (id)  /* zero when weak global ref is desired */
        {
            lock locked;

            for (std::multimap<int, countedRef>::iterator iter = refs.find(id);
                 iter != refs.end();
                 iter++) {
                if (iter->first != id)
                    break;
                if (isSame(obj, iter->second.global))
                {
                    /* If it's in the table but not the same reference,
                     * it must be a local reference and must be deleted.
                     */
                    if (obj != iter->second.global)
                        get_vm_env()->DeleteLocalRef(obj);
                        
                    iter->second.count += 1;
                    return iter->second.global;
                }
            }

            JNIEnv *vm_env = get_vm_env();
            countedRef ref;

            ref.global = vm_env->NewGlobalRef(obj);
            ref.count = 1;
            refs.insert(std::pair<const int, countedRef>(id, ref));
            vm_env->DeleteLocalRef(obj);

            return ref.global;
        }
        else
            return (jobject) get_vm_env()->NewWeakGlobalRef(obj);
    }

    return NULL;
}

jobject JCCEnv::deleteGlobalRef(jobject obj, int id)
{
    if (obj)
    {
        if (id)  /* zero when obj is weak global ref */
        {
            lock locked;

            for (std::multimap<int, countedRef>::iterator iter = refs.find(id);
                 iter != refs.end();
                 iter++) {
                if (iter->first != id)
                    break;
                if (isSame(obj, iter->second.global))
                {
                    if (iter->second.count == 1)
                    {
                        JNIEnv *vm_env = get_vm_env();

                        if (!vm_env)
                        {
                            /* Python's cyclic garbage collector may remove
                             * an object inside a thread that is not attached
                             * to the JVM. This makes sure the JVM doesn't
                             * segfault.
                             */
                            attachCurrentThread(NULL, 0);
                            vm_env = get_vm_env();
                        }

                        vm_env->DeleteGlobalRef(iter->second.global);
                        refs.erase(iter);
                    }
                    else
                        iter->second.count -= 1;

                    return NULL;
                }
            }

            printf("deleting non-existent ref: 0x%x\n", id);
        }
        else
            get_vm_env()->DeleteWeakGlobalRef((jweak) obj);
    }

    return NULL;
}

jclass JCCEnv::getClass(getclassfn initializeClass) const
{
    jclass cls = (*initializeClass)(true);

    if (cls == NULL)
    {
        lock locked;
        cls = (*initializeClass)(false);
    }

    return cls;
}

jobject JCCEnv::newObject(getclassfn initializeClass, jmethodID **mids,
                          int m, ...)
{
    jclass cls = getClass(initializeClass);
    JNIEnv *vm_env = get_vm_env();
    jobject obj;

    if (vm_env)
    {
        va_list ap;

        va_start(ap, m);
        obj = vm_env->NewObjectV(cls, (*mids)[m], ap);
        va_end(ap);
    }
#ifdef PYTHON
    else
    {
        PythonGIL gil;

        PyErr_SetString(PyExc_RuntimeError, "attachCurrentThread() must be called first");
        throw _EXC_PYTHON;
    }
#else
    else
        throw _EXC_JAVA;
#endif

    reportException();

    return obj;
}

jobjectArray JCCEnv::newObjectArray(jclass cls, int size)
{
    jobjectArray array = get_vm_env()->NewObjectArray(size, cls, NULL);

    reportException();
    return array;
}

void JCCEnv::setObjectArrayElement(jobjectArray array, int n,
                                   jobject obj) const
{
    get_vm_env()->SetObjectArrayElement(array, n, obj);
    reportException();
}

jobject JCCEnv::getObjectArrayElement(jobjectArray array, int n) const
{
    jobject obj = get_vm_env()->GetObjectArrayElement(array, n);

    reportException();
    return obj;
}

int JCCEnv::getArrayLength(jarray array) const
{
    int len = get_vm_env()->GetArrayLength(array);

    reportException();
    return len;
}

#ifdef PYTHON
jclass JCCEnv::getPythonExceptionClass() const
{
    return _thr;
}
#endif

void JCCEnv::reportException() const
{
    JNIEnv *vm_env = get_vm_env();
    jthrowable throwable = vm_env->ExceptionOccurred();

    if (throwable)
    {
        if (!env->handlers)
            vm_env->ExceptionDescribe();

#ifdef PYTHON
        PythonGIL gil;

        if (PyErr_Occurred())
        {
            /* _thr is PythonException ifdef _jcc_lib (shared mode)
             * if not shared mode, _thr is RuntimeException
             */
            jobject cls = (jobject) vm_env->GetObjectClass(throwable);

            if (vm_env->IsSameObject(cls, _thr))
            {
#ifndef _jcc_lib
                /* PythonException class is not available without shared mode.
                 * Python exception information thus gets lost and exception
                 * is reported via plain Java RuntimeException.
                 */
                PyErr_Clear();
                throw _EXC_JAVA;
#else
                throw _EXC_PYTHON;
#endif
            }
        }
#endif

        throw _EXC_JAVA;
    }
}


#define DEFINE_CALL(jtype, Type)                                         \
    jtype JCCEnv::call##Type##Method(jobject obj,                        \
                                     jmethodID mid, ...) const           \
    {                                                                    \
        va_list ap;                                                      \
        jtype result;                                                    \
                                                                         \
        va_start(ap, mid);                                               \
        result = get_vm_env()->Call##Type##MethodV(obj, mid, ap);        \
        va_end(ap);                                                      \
                                                                         \
        reportException();                                               \
                                                                         \
        return result;                                                   \
    }

#define DEFINE_NONVIRTUAL_CALL(jtype, Type)                              \
    jtype JCCEnv::callNonvirtual##Type##Method(jobject obj, jclass cls,  \
                                               jmethodID mid, ...) const \
    {                                                                    \
        va_list ap;                                                      \
        jtype result;                                                    \
                                                                         \
        va_start(ap, mid);                                               \
        result = get_vm_env()->CallNonvirtual##Type##MethodV(obj, cls,   \
                                                             mid, ap);   \
        va_end(ap);                                                      \
                                                                         \
        reportException();                                               \
                                                                         \
        return result;                                                   \
    }

#define DEFINE_STATIC_CALL(jtype, Type)                                 \
    jtype JCCEnv::callStatic##Type##Method(jclass cls,                  \
                                           jmethodID mid, ...) const    \
    {                                                                   \
        va_list ap;                                                     \
        jtype result;                                                   \
                                                                        \
        va_start(ap, mid);                                              \
        result = get_vm_env()->CallStatic##Type##MethodV(cls, mid, ap); \
        va_end(ap);                                                     \
                                                                        \
        reportException();                                              \
                                                                        \
        return result;                                                  \
    }
        
DEFINE_CALL(jobject, Object)
DEFINE_CALL(jboolean, Boolean)
DEFINE_CALL(jbyte, Byte)
DEFINE_CALL(jchar, Char)
DEFINE_CALL(jdouble, Double)
DEFINE_CALL(jfloat, Float)
DEFINE_CALL(jint, Int)
DEFINE_CALL(jlong, Long)
DEFINE_CALL(jshort, Short)

DEFINE_NONVIRTUAL_CALL(jobject, Object)
DEFINE_NONVIRTUAL_CALL(jboolean, Boolean)
DEFINE_NONVIRTUAL_CALL(jbyte, Byte)
DEFINE_NONVIRTUAL_CALL(jchar, Char)
DEFINE_NONVIRTUAL_CALL(jdouble, Double)
DEFINE_NONVIRTUAL_CALL(jfloat, Float)
DEFINE_NONVIRTUAL_CALL(jint, Int)
DEFINE_NONVIRTUAL_CALL(jlong, Long)
DEFINE_NONVIRTUAL_CALL(jshort, Short)

DEFINE_STATIC_CALL(jobject, Object)
DEFINE_STATIC_CALL(jboolean, Boolean)
DEFINE_STATIC_CALL(jbyte, Byte)
DEFINE_STATIC_CALL(jchar, Char)
DEFINE_STATIC_CALL(jdouble, Double)
DEFINE_STATIC_CALL(jfloat, Float)
DEFINE_STATIC_CALL(jint, Int)
DEFINE_STATIC_CALL(jlong, Long)
DEFINE_STATIC_CALL(jshort, Short)

void JCCEnv::callVoidMethod(jobject obj, jmethodID mid, ...) const
{
    va_list ap;

    va_start(ap, mid);
    get_vm_env()->CallVoidMethodV(obj, mid, ap);
    va_end(ap);

    reportException();
}

void JCCEnv::callNonvirtualVoidMethod(jobject obj, jclass cls,
                                      jmethodID mid, ...) const
{
    va_list ap;

    va_start(ap, mid);
    get_vm_env()->CallNonvirtualVoidMethodV(obj, cls, mid, ap);
    va_end(ap);

    reportException();
}

void JCCEnv::callStaticVoidMethod(jclass cls, jmethodID mid, ...) const
{
    va_list ap;

    va_start(ap, mid);
    get_vm_env()->CallStaticVoidMethodV(cls, mid, ap);
    va_end(ap);

    reportException();
}


jboolean JCCEnv::booleanValue(jobject obj) const
{
    return get_vm_env()->CallBooleanMethod(obj, _mids[mid_Boolean_booleanValue]);
}

jbyte JCCEnv::byteValue(jobject obj) const
{
    return get_vm_env()->CallByteMethod(obj, _mids[mid_Byte_byteValue]);
}

jchar JCCEnv::charValue(jobject obj) const
{
    return get_vm_env()->CallCharMethod(obj, _mids[mid_Character_charValue]);
}

jdouble JCCEnv::doubleValue(jobject obj) const
{
    return get_vm_env()->CallDoubleMethod(obj, _mids[mid_Double_doubleValue]);
}

jfloat JCCEnv::floatValue(jobject obj) const
{
    return get_vm_env()->CallFloatMethod(obj, _mids[mid_Float_floatValue]);
}

jint JCCEnv::intValue(jobject obj) const
{
    return get_vm_env()->CallIntMethod(obj, _mids[mid_Integer_intValue]);
}

jlong JCCEnv::longValue(jobject obj) const
{
    return get_vm_env()->CallLongMethod(obj, _mids[mid_Long_longValue]);
}

jshort JCCEnv::shortValue(jobject obj) const
{
    return get_vm_env()->CallShortMethod(obj, _mids[mid_Short_shortValue]);
}

jobject JCCEnv::boxBoolean(jboolean value) const
{
    return get_vm_env()->NewObject(_boo, _mids[mid_Boolean_init], value);
}

jobject JCCEnv::boxByte(jbyte value) const
{
    return get_vm_env()->NewObject(_byt, _mids[mid_Byte_init], value);
}

jobject JCCEnv::boxChar(jchar value) const
{
    return get_vm_env()->NewObject(_cha, _mids[mid_Character_init], value);
}

jobject JCCEnv::boxDouble(jdouble value) const
{
    return get_vm_env()->NewObject(_dou, _mids[mid_Double_init], value);
}

jobject JCCEnv::boxFloat(jfloat value) const
{
    return get_vm_env()->NewObject(_flo, _mids[mid_Float_init], value);
}

jobject JCCEnv::boxInteger(jint value) const
{
    return get_vm_env()->NewObject(_int, _mids[mid_Integer_init], value);
}

jobject JCCEnv::boxLong(jlong value) const
{
    return get_vm_env()->NewObject(_lon, _mids[mid_Long_init], value);
}

jobject JCCEnv::boxShort(jshort value) const
{
    return get_vm_env()->NewObject(_sho, _mids[mid_Short_init], value);
}


jmethodID JCCEnv::getMethodID(jclass cls, const char *name,
                              const char *signature) const
{
    jmethodID id = get_vm_env()->GetMethodID(cls, name, signature);

    reportException();

    return id;
}

jfieldID JCCEnv::getFieldID(jclass cls, const char *name,
                            const char *signature) const
{
    jfieldID id = get_vm_env()->GetFieldID(cls, name, signature);

    reportException();

    return id;
}


jmethodID JCCEnv::getStaticMethodID(jclass cls, const char *name,
                                    const char *signature) const
{
    jmethodID id = get_vm_env()->GetStaticMethodID(cls, name, signature);

    reportException();

    return id;
}

jobject JCCEnv::getStaticObjectField(jclass cls, const char *name,
                                     const char *signature) const
{
    JNIEnv *vm_env = get_vm_env();
    jfieldID id = vm_env->GetStaticFieldID(cls, name, signature);

    reportException();

    return vm_env->GetStaticObjectField(cls, id);
}

#define DEFINE_GET_STATIC_FIELD(jtype, Type, signature)                 \
    jtype JCCEnv::getStatic##Type##Field(jclass cls,                    \
                                         const char *name) const        \
    {                                                                   \
        JNIEnv *vm_env = get_vm_env();                                  \
        jfieldID id = vm_env->GetStaticFieldID(cls, name, #signature);  \
        reportException();                                              \
        return vm_env->GetStatic##Type##Field(cls, id);                 \
    }

DEFINE_GET_STATIC_FIELD(jboolean, Boolean, Z)
DEFINE_GET_STATIC_FIELD(jbyte, Byte, B)
DEFINE_GET_STATIC_FIELD(jchar, Char, C)
DEFINE_GET_STATIC_FIELD(jdouble, Double, D)
DEFINE_GET_STATIC_FIELD(jfloat, Float, F)
DEFINE_GET_STATIC_FIELD(jint, Int, I)
DEFINE_GET_STATIC_FIELD(jlong, Long, J)
DEFINE_GET_STATIC_FIELD(jshort, Short, S)

#define DEFINE_GET_FIELD(jtype, Type)                                   \
    jtype JCCEnv::get##Type##Field(jobject obj, jfieldID id) const      \
    {                                                                   \
        jtype value = get_vm_env()->Get##Type##Field(obj, id);          \
        reportException();                                              \
        return value;                                                   \
    }

DEFINE_GET_FIELD(jobject, Object)
DEFINE_GET_FIELD(jboolean, Boolean)
DEFINE_GET_FIELD(jbyte, Byte)
DEFINE_GET_FIELD(jchar, Char)
DEFINE_GET_FIELD(jdouble, Double)
DEFINE_GET_FIELD(jfloat, Float)
DEFINE_GET_FIELD(jint, Int)
DEFINE_GET_FIELD(jlong, Long)
DEFINE_GET_FIELD(jshort, Short)

#define DEFINE_SET_FIELD(jtype, Type)                                   \
    void JCCEnv::set##Type##Field(jobject obj, jfieldID id,             \
                                  jtype value) const                    \
    {                                                                   \
        get_vm_env()->Set##Type##Field(obj, id, value);                 \
        reportException();                                              \
    }

DEFINE_SET_FIELD(jobject, Object)
DEFINE_SET_FIELD(jboolean, Boolean)
DEFINE_SET_FIELD(jbyte, Byte)
DEFINE_SET_FIELD(jchar, Char)
DEFINE_SET_FIELD(jdouble, Double)
DEFINE_SET_FIELD(jfloat, Float)
DEFINE_SET_FIELD(jint, Int)
DEFINE_SET_FIELD(jlong, Long)
DEFINE_SET_FIELD(jshort, Short)

void JCCEnv::setClassPath(const char *classPath)
{
    JNIEnv *vm_env = get_vm_env();
    jclass _ucl = (jclass) vm_env->FindClass("java/net/URLClassLoader");
    jclass _fil = (jclass) vm_env->FindClass("java/io/File");
    jmethodID mid = vm_env->GetStaticMethodID(_ucl, "getSystemClassLoader",
                                              "()Ljava/lang/ClassLoader;");
    jobject classLoader = vm_env->CallStaticObjectMethod(_ucl, mid);
    jmethodID mf = vm_env->GetMethodID(_fil, "<init>", "(Ljava/lang/String;)V");
    jmethodID mu = vm_env->GetMethodID(_fil, "toURL", "()Ljava/net/URL;");
    jmethodID ma = vm_env->GetMethodID(_ucl, "addURL", "(Ljava/net/URL;)V");
#if defined(_MSC_VER) || defined(__WIN32)
    const char *pathsep = ";";
    char *path = _strdup(classPath);
#else
    const char *pathsep = ":";
    char *path = strdup(classPath);
#endif

    for (char *cp = strtok(path, pathsep);
         cp != NULL;
         cp = strtok(NULL, pathsep)) {
        jstring string = vm_env->NewStringUTF(cp);
        jobject file = vm_env->NewObject(_fil, mf, string);
        jobject url = vm_env->CallObjectMethod(file, mu);

        vm_env->CallVoidMethod(classLoader, ma, url);
    }
    free(path);
}

char *JCCEnv::getClassPath()
{
    JNIEnv *vm_env = get_vm_env();
    jclass _ucl = (jclass) vm_env->FindClass("java/net/URLClassLoader");
    jclass _url = (jclass) vm_env->FindClass("java/net/URL");
    jmethodID mid = vm_env->GetStaticMethodID(_ucl, "getSystemClassLoader",
                                              "()Ljava/lang/ClassLoader;");
    jobject classLoader = vm_env->CallStaticObjectMethod(_ucl, mid);
    jmethodID gu = vm_env->GetMethodID(_ucl, "getURLs", "()[Ljava/net/URL;");
    jmethodID gp = vm_env->GetMethodID(_url, "getPath", "()Ljava/lang/String;");
#if defined(_MSC_VER) || defined(__WIN32)
    const char *pathsep = ";";
#else
    const char *pathsep = ":";
#endif
    jobjectArray array = (jobjectArray)
        vm_env->CallObjectMethod(classLoader, gu);
    int count = array ? vm_env->GetArrayLength(array) : 0;
    int first = 1, total = 0;
    char *classpath = NULL;
    
    for (int i = 0; i < count; i++) {
        jobject url = vm_env->GetObjectArrayElement(array, i);
        jstring path = (jstring) vm_env->CallObjectMethod(url, gp);
        const char *chars = vm_env->GetStringUTFChars(path, NULL);
        int size = vm_env->GetStringUTFLength(path);

        total += size + 1;
        if (classpath == NULL)
            classpath = (char *) calloc(total, 1);
        else
            classpath = (char *) realloc(classpath, total);
        if (classpath == NULL)
            return NULL;

        if (first)
            first = 0;
        else
            strcat(classpath, pathsep);

        strcat(classpath, chars);
    }

    return classpath;
}

jstring JCCEnv::fromUTF(const char *bytes) const
{
    jstring str = get_vm_env()->NewStringUTF(bytes);

    reportException();

    return str;
}

char *JCCEnv::toUTF(jstring str) const
{
    JNIEnv *vm_env = get_vm_env();
    int len = vm_env->GetStringUTFLength(str);
    char *bytes = new char[len + 1];
    jboolean isCopy = 0;
    const char *utf = vm_env->GetStringUTFChars(str, &isCopy);

    if (!bytes)
        return NULL;

    memcpy(bytes, utf, len);
    bytes[len] = '\0';

    vm_env->ReleaseStringUTFChars(str, utf);

    return bytes;
}

char *JCCEnv::toString(jobject obj) const
{
    try {
        return obj
            ? toUTF((jstring) callObjectMethod(obj, _mids[mid_obj_toString]))
            : NULL;
    } catch (int e) {
        switch (e) {
          case _EXC_PYTHON:
            return NULL;
          case _EXC_JAVA: {
              JNIEnv *vm_env = get_vm_env();

              vm_env->ExceptionDescribe();
              vm_env->ExceptionClear();

              return NULL;
          }
          default:
            throw;
        }
    }
}

char *JCCEnv::getClassName(jobject obj) const
{
    return obj
        ? toString(callObjectMethod(obj, _mids[mid_obj_getClass]))
        : NULL;
}

#ifdef PYTHON

jstring JCCEnv::fromPyString(PyObject *object) const
{
    if (object == Py_None)
        return NULL;

    if (PyUnicode_Check(object))
    {
        if (sizeof(Py_UNICODE) == sizeof(jchar))
        {
            jchar *buf = (jchar *) PyUnicode_AS_UNICODE(object);
            jsize len = (jsize) PyUnicode_GET_SIZE(object);

            return get_vm_env()->NewString(buf, len);
        }
        else
        {
            jsize len = PyUnicode_GET_SIZE(object);
            Py_UNICODE *pchars = PyUnicode_AS_UNICODE(object);
            jchar *jchars = new jchar[len];
            jstring str;

            for (int i = 0; i < len; i++)
                jchars[i] = (jchar) pchars[i];

            str = get_vm_env()->NewString(jchars, len);
            delete jchars;

            return str;
        }
    }
    else if (PyString_Check(object))
        return fromUTF(PyString_AS_STRING(object));
    else
    {
        PyObject *tuple = Py_BuildValue("(sO)", "expected a string", object);

        PyErr_SetObject(PyExc_TypeError, tuple);
        Py_DECREF(tuple);

        return NULL;
    }
}

PyObject *JCCEnv::fromJString(jstring js, int delete_local_ref) const
{
    if (!js)
        Py_RETURN_NONE;

    JNIEnv *vm_env = get_vm_env();
    PyObject *string;

    if (sizeof(Py_UNICODE) == sizeof(jchar))
    {
        jboolean isCopy;
        const jchar *buf = vm_env->GetStringChars(js, &isCopy);
        jsize len = vm_env->GetStringLength(js);

        string = PyUnicode_FromUnicode((const Py_UNICODE *) buf, len);
        vm_env->ReleaseStringChars(js, buf);
    }
    else
    {
        jsize len = vm_env->GetStringLength(js);

        string = PyUnicode_FromUnicode(NULL, len);
        if (string)
        {
            jboolean isCopy;
            const jchar *jchars = vm_env->GetStringChars(js, &isCopy);
            Py_UNICODE *pchars = PyUnicode_AS_UNICODE(string);

            for (int i = 0; i < len; i++)
                pchars[i] = (Py_UNICODE) jchars[i];
        
            vm_env->ReleaseStringChars(js, jchars);
        }
    }

    if (delete_local_ref)
        vm_env->DeleteLocalRef((jobject) js);

    return string;
}


/* may be called from finalizer thread which has no vm_env thread local */
void JCCEnv::finalizeObject(JNIEnv *jenv, PyObject *obj)
{
    PythonGIL gil;

    set_vm_env(jenv);
    Py_DECREF(obj);
}

#endif /* PYTHON */
