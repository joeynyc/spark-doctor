from spark_doctor.collectors.gpu import _parse_dmon, peak_sample

DMON_SAMPLE = """\
# gpu   pwr gtemp mtemp   sm  mem  enc  dec  mclk  pclk
# Idx     W     C     C    %    %    %    %   MHz   MHz
    0    11    40     -    2    0    -    -  2411  2411
    0    12    41     -    5    1    -    -  2411  2411
    0    63    50     -   95   20    -    -  2411  2502
"""


def test_parse_dmon_extracts_rows():
    samples = _parse_dmon(DMON_SAMPLE)
    assert len(samples) == 3
    s = samples[0]
    assert s.gpu_power_draw_watts == 11
    assert s.gpu_temperature_c == 40
    assert s.gpu_utilization_percent == 2
    assert s.gpu_clock_mhz == 2411
    assert s.gpu_memory_clock_mhz == 2411


def test_parse_dmon_handles_na_dash():
    # mtemp is '-' in the sample; should not crash
    samples = _parse_dmon(DMON_SAMPLE)
    assert all(s.gpu_utilization_percent is not None for s in samples)


def test_peak_sample_picks_highest_util():
    samples = _parse_dmon(DMON_SAMPLE)
    peak = peak_sample(samples)
    assert peak is not None
    assert peak.gpu_utilization_percent == 95
    assert peak.gpu_power_draw_watts == 63
