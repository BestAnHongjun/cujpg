#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>
#include <string>
#include <stdint.h>


class Timer
{
private:
    std::string timer_name;
    uint32_t output_freq;
    double total_time = 0.0;
    double times = 0;
    double start_time = 0.0;


public:
    Timer(std::string name, uint32_t output_freq);
    ~Timer();

    void start();
    void end();


private:
    static double cpuSecond();
    void output_timer_info();
};


Timer::Timer(std::string name, uint32_t output_freq=0)
{
    this->timer_name = name;
    this->output_freq = output_freq;
}

double Timer::cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void Timer::start()
{
    this->start_time = cpuSecond();
}

void Timer::end()
{
    double interval = cpuSecond() - this->start_time;
    this->times += 1;
    this->total_time += interval;
    if ((this->output_freq) && ((uint64_t)this->times % (uint64_t)this->output_freq == 0))
    {
        output_timer_info();
    }
}

void Timer::output_timer_info()
{
    printf("[%s]total_time:%.4lf ms, cnt:%.0lf, avg_time:%.4lf ms\n", 
        this->timer_name.c_str(), this->total_time, this->times, this->total_time / this->times);
    return;
}

Timer::~Timer()
{
    printf("[Summary]");
    output_timer_info();
}


#endif  // __TIMER_H__