# Water Meter Server

This pytho code library receives data from field devices and disperses to appropriate directory on skylark.corp.janusresearch.com.

## Testing

Open terminal and execute BASH code:

```
pi@raspberrypi:~$ sudo python3 /opt/Janus/DATA/python3/janusdata.py
```

## Operational Execution

The script starts automatically when Skylark boots, or alternatively, open terminal and execute BASH code:

```
pi@raspberrypi:~$ sudo systemctl start janusdata
```

The status of the service can be checked with:

```
pi@raspberrypi:~$ sudo systemctl status janusdata
```

To stop the service, open terminal and execute BASH code:

```
pi@raspberrypi:~$ sudo systemctl stop janusdata
```

Data can be viewed in the following directory on Skylark:

```
/opt/Janus/datafiles/<device name>/
```
