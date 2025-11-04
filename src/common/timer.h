#pragma once

#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <string>

#include "general.h"
#include "logger.h"


#ifdef PE_USE_CUDA
#include <helper_timer.h>
#endif

class Timer {
  public:
    Timer(){};
    virtual ~Timer(){};
    virtual void tik(std::ostream& os, std::string) = 0;
    virtual double tok(std::ostream& os, std::string) = 0;
};

#ifdef PE_USE_OMP
#include <omp.h>

class OmpTimer : public Timer {
  public:
    OmpTimer(){};
    virtual ~OmpTimer(){};
    virtual void tik(std::ostream& os, std::string label) {
        auto iter = m_map.find(label);
        if (iter != m_map.end()) {
            iter->second = (double)omp_get_wtime();
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] start" << std::endl;
            os << "[" << label << "] TIK" << std::endl;
        } else {
            double ts = (double)omp_get_wtime();
            m_map.insert(std::pair<std::string, double>(label, ts));
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] start" << std ::endl;
            os << "[" << label << "] TIK" << std ::endl;
        }
    };
    virtual double tok(std::ostream& os, std::string label) {
        auto iter = m_map.find(label);
        if (iter != m_map.end()) {
            double ts_old = iter->second;
            iter->second = (double)omp_get_wtime();
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] end interval = " <<
            // std::showpoint << iter->second - ts_old << "s" << std::endl;
            os << "[" << label << "] TOK Dt = " << std::showpoint << iter->second - ts_old << "s" << std::endl;
            return iter->second - ts_old;
        } else {
            // LOG_OSTREAM_WARN << "Timer ["<< label << "] not found" <<
            // std::endl;
            os << "[" << label << "] not found" << std::endl;
            return -1;
        }
    };
    virtual void clear() { m_map.clear(); }

  protected:
    std::map<std::string, double> m_map;
};

#endif

class BasicTimer : public Timer {
  public:
    BasicTimer(){};
    virtual ~BasicTimer(){};
    virtual void tik(std::ostream& os, std::string label) override {
        auto iter = m_map.find(label);
        if (iter != m_map.end()) {
            iter->second = std::chrono::steady_clock::now();
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] start" << std::endl;
            os << "[" << label << "] TIK" << std::endl;
        } else {
            auto ts = std::chrono::steady_clock::now();
            m_map.insert(std::make_pair(label, ts));
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] start" << std ::endl;
            os << "[" << label << "] TIK" << std ::endl;
        }
    }
    virtual double tok(std::ostream& os, std::string label) override {
        auto iter = m_map.find(label);
        if (iter != m_map.end()) {
            auto ts_old = iter->second;
            iter->second = std::chrono::steady_clock::now();
            // LOG_OSTREAM_INFO << "Timer ["<< label << "] end interval = " <<
            // std::showpoint << iter->second - ts_old << "s" << std::endl;
            auto duration_in_seconds = (iter->second - ts_old).count() / 1000000000.0;
            os << "[" << label << "] TOK Dt = " << std::showpoint << duration_in_seconds << "s" << std::endl;
            return duration_in_seconds;
        } else {
            // LOG_OSTREAM_WARN << "Timer ["<< label << "] not found" <<
            // std::endl;
            os << "[" << label << "] not found" << std::endl;
            return -1;
        }
    }
    virtual void clear() { m_map.clear(); }

  protected:
    std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>> m_map;
};

// ---------- Benchmark Timer -------------- //

class BenchmarkTimer {
  private:
    struct _Timer {
        bool running;
        double time;
        int invocations;

        #ifdef PE_USE_CUDA
        StopWatchInterface *timer;
        _Timer() : running(false), time(0), invocations(0) { sdkCreateTimer(&timer); start(); }
        // double duration() { return timer.(std::chrono::steady_clock::now() - startTime).count() / 1000000000.0; }
        double duration() { return sdkGetTimerValue(&timer) / 1000.0; }
        
        // gets elapsed time (even if currently running)
        double elapsed() { return running ? time + duration() : time; }
        void stop() {
            assert(running);
            sdkStopTimer(&timer);
            time += duration();
            running = false;
            sdkResetTimer(&timer);
        }
        void start() {
            if (running) {
                std::cerr << "ERROR: timer already running. Reported timings will be inaccurate." << std::endl;
                stop();
            }
            assert(!running);
            running = true;
            ++invocations;
            // startTime = std::chrono::steady_clock::now();
            sdkStartTimer(&timer);
        }

        #else
        std::chrono::time_point<std::chrono::steady_clock> startTime;
        _Timer() : running(false), time(0), invocations(0) { start(); }
        double duration() { return (std::chrono::steady_clock::now() - startTime).count() / 1000000000.0; }
        // gets elapsed time (even if currently running)
        double elapsed() { return running ? time + duration() : time; }
        void stop() {
            assert(running);
            time += duration();
            running = false;
        }
        void start() {
            if (running) {
                std::cerr << "ERROR: timer already running. Reported timings will be inaccurate." << std::endl;
                stop();
            }
            assert(!running);
            running = true;
            ++invocations;
            startTime = std::chrono::steady_clock::now();
        }
        #endif
    };

    typedef std::map<std::string, _Timer> TimerMap;

    struct _Section : public _Timer {
        TimerMap timers;
        _Section() : _Timer() {}
        void startTimer(std::string name) {
            auto it = timers.find(name);
            if (it != timers.end()) {
                if (it->second.running) {
                    std::cerr << "ERROR: timer " << name << " already started. " << std::endl;
                }
                it->second.start();
            } else {
                timers[name] = _Timer();
            }
        }

        using _Timer::duration;
        using _Timer::start;
        void stop() {
            _Timer::stop();
            // Also stop all our sub-timers...
            for (auto& entry : timers) {
                if (entry.second.running) {
                    std::cerr << "WARNING: stopping timer " << entry.first
                              << " implicitly in enclosing section's stop()" << std::endl;
                    entry.second.stop();
                }
            }
        }

        void start(const std::string& name) {
            auto lb = timers.lower_bound(name);
            if ((lb == timers.end()) || (lb->first != name))
                timers.emplace_hint(lb, name, _Timer());
            else
                lb->second.start();  // The full section timer must be started too...
        }
        void stop(const std::string& name) { timers.at(name).stop(); }

        void report(std::ostream& os) {
            for (auto& entry : timers)
                os << displayName(entry.first) << '\t' << entry.second.elapsed() << '\t' << entry.second.invocations
                   << std::endl;
        }
    };

    typedef std::map<std::string, _Section> SectionMap;
    typedef SectionMap::iterator SectionIterator;
    typedef SectionMap::const_iterator SectionConstIterator;

    SectionMap m_sections;
    std::list<std::string> m_sectionStack;

    static std::string displayName(std::string name) {
        size_t levels = 0;
        for (char c : name)
            if (c == ':') ++levels;
        if (levels == 0) return "-> " + name;

        std::string result(4 * levels, ' ');
        result += "|-> ";
        result.append(name, name.rfind(':') + 1, std::string::npos);
        return result;
    }

  public:
    BenchmarkTimer() { reset(); }

    void startSection(const std::string& name, bool verbose = false) {
        std::string label;
        if (!m_sectionStack.empty())
            label = m_sectionStack.back() + ':' + name;
        else
            label = name;
        m_sectionStack.push_back(label);
        auto lb = m_sections.lower_bound(label);
        if ((lb == m_sections.end()) || (lb->first != label))
            m_sections.emplace_hint(lb, label, _Section());
        else
            lb->second.start();  // The full section timer must be started too...

        if (verbose) std::cout << "[" << name << "] TIK" << std ::endl;
    }

    void stopSection(const std::string& name, bool verbose = false) {
        assert(!m_sectionStack.empty());
        std::string currentLabel = m_sectionStack.back();
        m_sectionStack.pop_back();

        std::string label;
        if (!m_sectionStack.empty())
            label = m_sectionStack.back() + ':' + name;
        else
            label = name;

        if (label != currentLabel) {
            std::cerr << "ERROR: sections must be stopped in the reverse of "
                         "the order they were started."
                      << std::endl;
            std::cerr << "(Expected " << currentLabel << ", but got " << label << ")" << std::endl;
        }

        if (verbose)
            std::cout << "[" << name << "] TOK Dt = " << std::showpoint << m_sections.at(label).duration() << "s"
                      << std::endl;
        m_sections.at(label).stop();
    }

    void start(std::string timer) {
        std::string sectionName;
        if (!m_sectionStack.empty()) {
            sectionName = m_sectionStack.back();
            timer = sectionName + ':' + timer;
        }

        m_sections.at(sectionName).start(timer);
    }

    void stop(std::string timer) {
        std::string sectionName;
        if (!m_sectionStack.empty()) {
            sectionName = m_sectionStack.back();
            timer = sectionName + ':' + timer;
        }

        m_sections.at(sectionName).stop(timer);
    }

    // Remove all timers and restart the global section
    void reset() {
        m_sections.clear();
        m_sectionStack.clear();
        // (re)start global section
        m_sections.insert(std::make_pair("", _Section()));
    }

    void report(std::ostream& os) {
        for (SectionIterator it = m_sections.begin(); it != m_sections.end(); ++it) {
            if (it->first != "") {  // Skip global section... this is reported at the end
                os << displayName(it->first) << "\t" << it->second.elapsed() << "\t" << it->second.invocations
                   << std::endl;
                it->second.report(os);
            }
        }
        // TODO add percentage

        m_sections.at("").report(os);
        os << "Full time\t" << m_sections.at("").elapsed() << std::endl;
        os << "========" << std::endl;
    }
};

extern std::unique_ptr<Timer> g_timer;

#ifdef GLOBALBENCHMARK
extern std::unique_ptr<BenchmarkTimer> g_benchmarkTimer;
#define BENCHMARK_START_TIMER_SECTION(label)         \
    {                                                \
        LOG_OSTREAM_DEBUG;                           \
        g_benchmarkTimer->startSection(label, true); \
    }
#define BENCHMARK_STOP_TIMER_SECTION(label)         \
    {                                               \
        LOG_OSTREAM_DEBUG;                          \
        g_benchmarkTimer->stopSection(label, true); \
    }
inline void BENCHMARK_RESET() { g_benchmarkTimer->reset(); }
inline void BENCHMARK_REPORT() { g_benchmarkTimer->report(std::cout); }
#else
#define BENCHMARK_START_TIMER_SECTION(label) ;
#define BENCHMARK_STOP_TIMER_SECTION(label) ;
inline void BENCHMARK_RESET() {}
inline void BENCHMARK_REPORT() {}
#endif

#define TIMER_TIK(label)                \
    {                                   \
        LOG_OSTREAM_DEBUG;              \
        g_timer->tik(std::cout, label); \
    }
#define TIMER_TOK(label)                \
    {                                   \
        LOG_OSTREAM_DEBUG;              \
        g_timer->tok(std::cout, label); \
    }
#define TIMER_CLEAR() g_timer->clear()

struct ScopedBenchmarkTimer {
    ScopedBenchmarkTimer(const std::string& name) : m_name(name) {
        g_benchmarkTimer->startSection(name, false);
    }

    ~ScopedBenchmarkTimer() { g_benchmarkTimer->stopSection(m_name, false); }

  private:
    std::string m_name;
};

#ifdef GLOBALBENCHMARK
#define PHY_PROFILE(label) ScopedBenchmarkTimer __profile(label)
#else
#define PHY_PROFILE(label)
#endif