"""Provides a wrapper for serial communication with Jena piezoelectric controllers
"""

import serial

class NV40:
    """Wrapper for serial communication with Jena NV 40/1 CL piezoelectric controller

    Args:
        port (str): serial port
        timeout (float, optional): serial timeout in seconds
        closed_loop (bool, optional): Operation mode. True for closed loop, False for open loop.
    
    Example use::

        device = jena.NV40('COM3')
        device.set_position(10)
        print(device.get_position())

    Recommended use is with a context manager, which restores manual control after executing the statements in the with block. Note that the position will be reset to the position it had before the remote control was switched on::

    setpoint = 100
    print('Setpoint',setpoint)
    with jena.NV40('COM3') as device:
        print('Initial position',device.get_position())
        device.set_position(setpoint)
        position = device.get_position()
        print('New position',device.get_position())
    print('Final position',device.get_position())

    Notes:
        ``python -m serial.tools.list_ports`` will print a list of available ports. It is also possible to add a regexp as first argument and the list will only include entries that matched. See https://pythonhosted.org/pyserial/shortintro.html#opening-serial-ports for more details.
    """
    def __init__(self, port, timeout=0.5, closed_loop=True):
        """Initialize device
        Set connection configuration, enable remote control and set to closed loop operation"""
        self.port = port
        self.timeout = timeout
        self.baudrate = 115200
        self.bytesize = serial.EIGHTBITS
        self.stopbits = serial.STOPBITS_ONE
        self.parity = serial.PARITY_NONE
        self.xonxoff = 0
        self.errors = {'err,1':'Unknown command',
                      'err,2':'Too many characters in the command',
                      'err,3':'Too many characters in the parameter',
                      'err,4':'Too many parameters',
                      'err,5':'Wrong character in parameter',
                      'err,6':'Wrong separator',
                      'err,7':'Position out of range'}
        self.set_remote_control(True)
        self.set_closed_loop(closed_loop)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Return to manual control after use with .. as statement"""
        self.set_remote_control(False)

    def __execute(self, command):
        """Execute a device command and check if the device returned an error
        
        Args:
            command (str): Device command
        
        Returns:
            str: Response from the device
        
        Raises:
            DeviceError
        """
        with serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize, 
            parity=self.parity, timeout=self.timeout) as ser:
            command += '\r'
            print('Executing command',command)
            ser.write(command.encode())
            answer = ser.read(100).decode("utf-8").rstrip()
            if answer in self.errors:
                raise ValueError(self.errors[answer])
            return answer
        
    def set_position(self, value):
        """Set position 

        Setting the position will enable remote control, which disables manual control
        You can return to manual control using the set_remote_control(False) method.
        
        Args:
            value (float): Position in urad/um (closed loop) or volts (open loop)
        """
        # self.set_remote_control(True)
        self.__execute('set,%.2f' % value)
        return self.get_position()
        """value 0-100 um"""
        
    def get_position(self):
        """Get current position
        
        Returns:
            float: Current position in urad or um (closed loop) or in volts (open loop)
        """
        return self.__execute('mess').split(',')[1]
        
    def set_closed_loop(self,is_enabled):
        """Switch between open and closed loop mode
        
        Args:
            is_enabled (bool): closed loop (True) or open loop (False) 
        """
        if is_enabled:
            self.__execute('cl,1')
        else:
            self.__execute('cl,0')
        
    def set_remote_control(self,is_enabled):
        """Switch between remote and manual control
        
        Args:
            is_enabled (bool): remote control (True) or manual control (False)
        """
        if is_enabled:
            self.__execute('fenable,1')
        else:
            self.__execute('fenable,0')