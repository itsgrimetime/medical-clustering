import sys

class Spinner:
    def __init__(self, chars=['\\', '|', '/', '-']):
	self.count = 0
	self.chars = chars

    def spin(self, num='', shownum=True, pct=True):
	sign = '%' if pct else ''
	numstr = "({}{})".format(num, sign) if shownum else ""
	status = "{}{}".format(self.chars[self.count % len(self.chars)], numstr)
	sys.stdout.write(status)
	sys.stdout.flush()
	sys.stdout.write('\b' * len(status))
	self.count += 1
