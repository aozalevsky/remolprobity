import sys

if sys.argv[1:2] == ['compare']:
    from .compare import main
    sys.exit(main(sys.argv[2:]))
if sys.argv[1:2] == ['stock']:
    from .stock import main
    sys.exit(main(sys.argv[2:]))
from .run import main

sys.exit(main())
