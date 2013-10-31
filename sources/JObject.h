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

#ifndef _JObject_H
#define _JObject_H

#include <stdio.h>
#include "JCCEnv.h"

class _DLL_EXPORT JObject {
public:
    jobject this$;

    inline explicit JObject(jobject obj)
    {
        if (obj)
        {
            id = env->id(obj);
            this$ = env->newGlobalRef(obj, id);
        }
        else
        {
            id = 0;
            this$ = NULL;
        }
    }

    inline JObject(const JObject& obj)
    {
        id = obj.id ? obj.id : env->id(obj.this$);
        this$ = env->newGlobalRef(obj.this$, id);
    }

    virtual ~JObject()
    {
        this$ = env->deleteGlobalRef(this$, id);
    }

    JObject& weaken$()
    {
        if (id)
        {
            jobject ref = env->newGlobalRef(this$, 0);

            env->deleteGlobalRef(this$, id);
            id = 0;
            this$ = ref;
        }

        return *this;
    }

    inline int operator!() const
    {
        return env->isSame(this$, NULL);
    }

    inline int operator==(const JObject& obj) const
    {
        return env->isSame(this$, obj.this$);
    }

    JObject& operator=(const JObject& obj)
    {
        jobject prev = this$;
        int objid = obj.id ? obj.id : env->id(obj.this$);

        this$ = env->newGlobalRef(obj.this$, objid);
        env->deleteGlobalRef(prev, id);
        id = objid;

        return *this;
    }

private:
    int id;      // zero when this$ is a weak ref */
};


#ifdef PYTHON

#include <Python.h>
#include "macros.h"

class t_JObject {
public:
    PyObject_HEAD
    JObject object;
};

extern PyTypeObject PY_TYPE(JObject);

#endif /* PYTHON */


#endif /* _JObject_H */
