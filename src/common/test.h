#pragma once

#include "math/math.h"
#include "common/logger.h"


ATTRIBUTE_ALIGNED16(class)
TestCase{
public:

    virtual std::string getName(){ return "DefaultCase"; }

    virtual bool judge(){ return true;}

    virtual void failCallback(){}

    virtual void test(){
        LOG_OSTREAM_INFO << "TestCase " << getName() << " begin:" << std::endl; 
        if(judge()) {
            LOG_OSTREAM_INFO << "==== pass ====" << std::endl; 
        } else {
            LOG_OSTREAM_INFO << "==== failed ====" << std::endl; 
            failCallback(); 
            exit(-1);
        }
    }
};
