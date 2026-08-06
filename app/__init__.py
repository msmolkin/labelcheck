"""labelcheck — TTB alcohol-label verification prototype.

Checks a label photograph against the health warning statement required by
27 CFR Part 16, and against the four application-record fields (brand,
class/type, alcohol content, net contents).

The package runs entirely offline: no module here opens a network socket, and
the systemd unit that hosts it sets ``IPAddressDeny=any``.
"""

__version__ = "0.1.0"
