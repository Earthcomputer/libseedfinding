import os
import sys

border = '// ----------------------------------------------------------------------------'

PythonVersion = sys.version_info[0];
# find all .h first
with open('lcg.h') as f:
    lines = f.readlines()
    inImplementation = False

    if PythonVersion < 3:
        os.makedirs('out')
    else:
        os.makedirs('out', exist_ok=True)

    with open('out/libseedfinding.h', 'w') as fh:
        with open('out/libseedfinding.cc', 'w') as fc:
            fc.write('#include "libseedfinding.h"\n')
            fc.write('namespace libseedfinding {\n')
            for line in lines:
                isBorderLine = border in line
                if isBorderLine:
                    inImplementation = not inImplementation
                else:
                    if inImplementation:
                        fc.write(line.replace('inline ', ''))
                        pass
                    else:
                        fh.write(line)
                        pass
            fc.write('} // namespace libseedfinding\n')
