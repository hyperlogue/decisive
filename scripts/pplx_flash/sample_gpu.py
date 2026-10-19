import argparse
import ctypes as c
import json
from pathlib import Path
import time


class GpuCounters:
    def __init__(self):
        cf = c.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        io = c.CDLL('/System/Library/Frameworks/IOKit.framework/IOKit')

        def api(lib, name, result, *arguments):
            function = getattr(lib, name)
            function.restype = result
            function.argtypes = arguments
            return function

        self.make_string = api(cf, 'CFStringCreateWithCString', c.c_void_p, c.c_void_p, c.c_char_p, c.c_uint32)
        self.release = api(cf, 'CFRelease', None, c.c_void_p)
        self.get_value = api(cf, 'CFDictionaryGetValue', c.c_void_p, c.c_void_p, c.c_void_p)
        self.number = api(cf, 'CFNumberGetValue', c.c_bool, c.c_void_p, c.c_int, c.c_void_p)
        self.string = api(cf, 'CFStringGetCString', c.c_bool, c.c_void_p, c.c_void_p, c.c_long, c.c_uint32)
        self.array_count = api(cf, 'CFArrayGetCount', c.c_long, c.c_void_p)
        self.array_value = api(cf, 'CFArrayGetValueAtIndex', c.c_void_p, c.c_void_p, c.c_long)
        self.property = api(io, 'IORegistryEntryCreateCFProperty', c.c_void_p, c.c_uint32, c.c_void_p, c.c_void_p, c.c_uint32)
        self.object_release = api(io, 'IOObjectRelease', c.c_int, c.c_uint32)
        matching = api(io, 'IOServiceMatching', c.c_void_p, c.c_char_p)(b'AGXAccelerator')
        self.service = api(io, 'IOServiceGetMatchingService', c.c_uint32, c.c_uint32, c.c_void_p)(0, matching)
        if not self.service:
            raise RuntimeError('No AGX GPU service')
        self.keys = {}
        self.clients = []
        self.fields = ['Device Utilization %', 'Renderer Utilization %', 'Tiler Utilization %',
                       'In use system memory', 'Alloc system memory', 'recoveryCount']
        iterator = c.c_uint32()
        children = api(io, 'IORegistryEntryGetChildIterator', c.c_int, c.c_uint32, c.c_char_p, c.c_void_p)
        next_child = api(io, 'IOIteratorNext', c.c_uint32, c.c_uint32)
        get_id = api(io, 'IORegistryEntryGetRegistryEntryID', c.c_int, c.c_uint32, c.c_void_p)
        if children(self.service, b'IOService', c.byref(iterator)) == 0:
            try:
                while child := next_child(iterator.value):
                    owner = self.property(child, self.key('IOUserClientCreator'), None, 0)
                    text = c.create_string_buffer(512)
                    selected = False
                    if owner:
                        try:
                            selected = self.string(owner, text, len(text), 0x08000100) and b'WebKit' in text.value
                        finally:
                            self.release(owner)
                    if selected:
                        identity = c.c_uint64()
                        get_id(child, c.byref(identity))
                        self.clients.append((child, str(identity.value)))
                    else:
                        self.object_release(child)
            finally:
                self.object_release(iterator.value)

    def key(self, name):
        if name not in self.keys:
            self.keys[name] = self.make_string(None, name.encode(), 0x08000100)
        return self.keys[name]

    def integer(self, dictionary, name):
        item = self.get_value(dictionary, self.key(name))
        if not item:
            return None
        value = c.c_int64()
        if not self.number(item, 4, c.byref(value)):
            return None
        return value.value

    def snapshot(self):
        started = time.time_ns()
        values = self.property(self.service, self.key('PerformanceStatistics'), None, 0)
        if not values:
            raise RuntimeError('GPU driver did not expose PerformanceStatistics')
        try:
            stats = {name: self.integer(values, name) for name in self.fields}
        finally:
            self.release(values)
        clients = {}
        for client, identity in self.clients:
            usage = self.property(client, self.key('AppUsage'), None, 0)
            if not usage:
                continue
            try:
                count = self.array_count(usage)
                total = sum(self.integer(self.array_value(usage, i), 'accumulatedGPUTime') or 0 for i in range(count))
                clients[identity] = {'webkit': True, 'accumulated_gpu_time': total, 'queues': count}
            finally:
                self.release(usage)
        ended = time.time_ns()
        return {'epoch_ms': (started + ended) / 2e6, 'sample_ms': (ended - started) / 1e6,
                'statistics': stats, 'clients': clients}

    def close(self):
        for client, _ in self.clients:
            self.object_release(client)
        self.clients.clear()
        if self.service:
            self.object_release(self.service)
            self.service = 0
        for key in self.keys.values():
            self.release(key)
        self.keys.clear()


def sample(output, duration, interval):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        counters = GpuCounters()
        try:
            started = time.monotonic()
            count = 0
            while time.monotonic() - started < duration:
                before = time.monotonic()
                stream.write(json.dumps(counters.snapshot()) + '\n')
                stream.flush()
                count += 1
                if count == 1:
                    print(json.dumps({'sampling': True, 'interval_seconds': interval}), flush=True)
                time.sleep(max(0, interval - (time.monotonic() - before)))
        finally:
            counters.close()
    print(json.dumps({'samples': count, 'elapsed_seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=20)
    parser.add_argument('--interval', type=float, default=.1)
    args = parser.parse_args()
    if not 0 < args.duration <= 300 or not .05 <= args.interval <= 5:
        parser.error('Duration must be 0–300 seconds; interval must be 0.05–5 seconds')
    sample(args.output, args.duration, args.interval)
