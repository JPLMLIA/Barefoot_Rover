// xts_2_ros_bridge.cpp : Captures data via XTS API and sends JSON requests to the ROS Bridge via websocket.

#include "pch.h"

#include <csignal>
#include <iostream>
#include <ctime>
#include <chrono>
#include <stdio.h>
#include <time.h>
#include <iomanip>

// Winsock2 has to be included before windows.h amidst any (eventual) Boost-related imports (which cpprest does).
// see https://stackoverflow.com/questions/38201102/including-boost-network-libraries-makes-windows-h-functions-undefined
#include <Winsock2.h>
#include <windows.h>

#include <XtsApi.h>
#include <nlohmann/json.hpp>
#include <cpprest/ws_client.h>
#include <libconfig.h++>

#include "matrix.h"

using namespace std;
using namespace std::chrono;
using namespace web;
using namespace web::websockets::client;
using namespace libconfig;
using json = nlohmann::json;

// struct to encapsulate handles and config for scan callback
struct ScanIOHandles {
	CMatrix<WORD> scanData;
	XtsDeviceHandle deviceHandle;
	websocket_client *wsClientPtr = NULL;
	ofstream *outputFilePtr = NULL;

	const char *ros_topic;
	const char *ros_message_type;

	/*
	 * GetScanData setting defaults, see XtsApi.h :
     *    mode: 0 - without zero correction and no pressure conversion
	 *    dataFormat : 1 - 16 bit strength
	 *    dataAcq : 0 - default data acquisition
	 */
	int mode = 0;
	int dataFormat = 1;
	int dataAcq = 0;

	Config *config = NULL;
};

HANDLE writerAckEvent;           // to perform graceful shutdowns on interrupt
bool isShutdown = FALSE;         // capture shutdown state
void SignalHandler(int signum);  // Signal Handler

// NOTE: XtsDeviceHandle's appear to be passed around by value in the sample code, so we'll do the same (type'd pointer?)

void UsageAndExit(const char *progName);
ScanIOHandles* InitHandles(Config *config, const char *deviceName, const char *bridgeEndpoint, const char *ros_topic, const char *ros_message_type, const char *outputFileName);
void CloseHandlesAndDelete(ScanIOHandles *ioHandles);
websocket_client* InitWSConnection(const char *bridgeEndpoint, const char *ros_topic, const char *ros_message_type);
void CloseWSConnection(websocket_client *wsClientPtr, const char *ros_topic);
void GetDataAndPublish(ScanIOHandles *ioHandles);
void __stdcall XtsCallbackFunction(XtsDeviceHandle deviceHandle, int status, DWORD frameNumber,	void *arg);
string MakeScanJSON(DWORD frameNumber, const char *ros_topic, CMatrix<WORD> &scanData);
string MakeScanJSON(DWORD frameNumber, const char *ros_topic, CMatrix<FLOAT> &scanData); 
void PublishFromFile(const char *fileName, const char *bridgeEndpoint, const char *ros_topic, const char *ros_message_type, const char *outputFileName);
void PublishMessage(websocket_client *wsClientPtr, string jsonMessage);
void WriteScanToFile(ostream &outputStream, DWORD frameNumber, CMatrix<WORD> &scanData);
void WriteScanToFile(ostream &outputStream, DWORD frameNumber, CMatrix<FLOAT> &scanData); 
void WriteRecordHeader(ostream &outputStream, DWORD frameNumber, int rowCount, int colCount);
void WriteCaptureStart(ostream &outputStream);
void WriteCaptureEnd(ostream &outputStream);
void OutputDateTime(ostream &outputStream, const char *formatString, bool appendMillis = FALSE);

#define MAX_PARAM_SIZE 64 // arbitrary but we have to pass allocated character arrays to INI files. That's Window's for you.

// configuration file and constants
#define CONFIG_FILE "xts_2_ros_bridge.cfg"

#define ROS_TOPIC_PARAM "xts_2_ros_bridge.ros.topic_name"
#define ROS_MSG_TYPE_PARAM "xts_2_ros_bridge.ros.message_type"
#define XTS_SCAN_MODE_PARAM "xts_2_ros_bridge.xts.mode"
#define XTS_SCAN_FORMAT_PARAM "xts_2_ros_bridge.xts.dataFormat"
#define XTS_SCAN_ACQ_PARAM "xts_2_ros_bridge.xts.dataAcq"

#define ROS_TOPIC_DEF "XtsBus"
#define ROS_MSG_TYPE_DEF "barefoot_rover_ros/XtsScan"

// #define DEBUG  // uncomment to see file ingest

/*-----------------------------------------------------------------------------------------------------------------------------
 * main()
 *-----------------------------------------------------------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
	cout << "Starting up..." << endl;

	if (argc < 3) UsageAndExit(argv[0]);

	Config *config = new Config();
	// can throw ConfigException
	config->readFile(CONFIG_FILE);

	int argNumber = 1;
	bool deviceRun = TRUE;
	char *outputFileName = NULL;
	char* bridgeEndpoint = NULL;
	if (strcmp(argv[argNumber], "-f") == 0)
	{
		if (argc < 4) UsageAndExit(argv[0]);
		deviceRun = FALSE;
		argNumber++;
	}
	else if (strcmp(argv[argNumber], "-o") == 0) {
		if (argc < 4) UsageAndExit(argv[0]);
		argNumber++;
		outputFileName = argv[argNumber++];
		cout << "Directing device output to file " << outputFileName << endl;
	}
	else if (strcmp(argv[argNumber], "-of") == 0) {
		// this option is for development testing only, being that it's not at all practical (i.e. from file to file). It is not advertised in usage above.
		if (argc < 4) UsageAndExit(argv[0]);
		deviceRun = FALSE;
		argNumber++;
		outputFileName = argv[argNumber++];
	}
	// whether this is a device id or file is implied by settings made above
	char* inputName = argv[argNumber++];

	if (outputFileName == NULL) {
		// routing output to ROS bridge
		bridgeEndpoint = argv[argNumber++];
		cout << "Using " << bridgeEndpoint << " as the ROS bridge endpoint." << endl;
	}
	else {
		cout << "Sending output to " << outputFileName << endl;
	}

	// initialize shutdown event : NULL - default security, TRUE - manual-reset, FALSE - initial state  = nonsignaled
	writerAckEvent = CreateEvent(NULL, TRUE, FALSE, TEXT("WriterShutdownAcknowledge"));

	// register signal SIGINT signal handler  
	signal(SIGINT, SignalHandler);

	// These are not used if output is going to file
	const char *ros_topic, *ros_message_type;
	if (!(config->lookupValue(ROS_TOPIC_PARAM, ros_topic))) {
		ros_topic = ROS_TOPIC_DEF;
		cerr << "Could not find " << ROS_TOPIC_PARAM << " in config, defaulting to " << ros_topic << endl;
	}
	else {
		cout << "Read topic " << ros_topic << " from config." << endl;
	}
	if (!(config->lookupValue(ROS_MSG_TYPE_PARAM, ros_message_type))) {
		ros_message_type = ROS_MSG_TYPE_DEF;
		cerr << "Could not find " << ROS_MSG_TYPE_PARAM << " in config, defaulting to " << ros_message_type << endl;
	}
	else {
		cout << "Read ros message type " << ros_message_type << " from config." << endl;
	}

	if (deviceRun)
	{
		// using the pressure pad
		cout << "Using " << inputName << " as the pressure pad device name." << endl;
        
		ScanIOHandles* ioHandles = InitHandles(config, inputName, bridgeEndpoint, ros_topic, ros_message_type, outputFileName);

		while (!isShutdown) {
			DWORD waitResult = WaitForSingleObject(writerAckEvent, INFINITE);
			cout << "Wait on shutdown event interrupted. (waitResult=" << waitResult << ")" << endl;
		}
		// we are finished here
		CloseHandlesAndDelete(ioHandles);
	}
	else {
		cout << "Obtaining input from file " << inputName << endl;

		// reading from a file
		PublishFromFile(inputName, bridgeEndpoint, ros_topic, ros_message_type, outputFileName);
	}

	// clean-up event object
	CloseHandle(writerAckEvent);
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * UsageAndExit()
 * Display syntax and exit
 *-----------------------------------------------------------------------------------------------------------------------------*/
void UsageAndExit(const char *progName) {
	cerr << "Usage : " << progName << " <Xts Device ID> <ROS Bridge WebSocket endpoint>      - standard operation : input from device, data routed to ROS bridge." << endl;
	cerr << "        " << progName << " -f <Input File Name> <ROS Bridge WebSocket endpoint> - input from file, data routed to ROS bridge." << endl;
	cerr << "        " << progName << " -o <Output File Name> <Xts Device ID>                - input from device, data written to local file." << endl;
	cerr << "Xts Device ID is typically          : 'VD0000000000001'" << endl;
	cerr << "ROS Bridge WS endpoint is typically : 'ws://localhost:9090'" << endl;
	cerr << "Input files must and output files will have same format as that produced by Xiroku's LLtest application." << endl;
	exit(0);
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * SignalHandler()
 * Signal handler for shutdowns on ctrl-c
 *-----------------------------------------------------------------------------------------------------------------------------*/
void SignalHandler(int signum) {
	cout << "Interrupt signal (" << signum << ") received, shutting down." << endl;

	isShutdown = TRUE;
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * InitHandles()
 * Open necessary connections, including handle
 * to device and output to webservice or file.
 *-----------------------------------------------------------------------------------------------------------------------------*/
ScanIOHandles* InitHandles(Config *config, const char *deviceName, const char *bridgeEndpoint, const char *ros_topic, const char *ros_message_type, const char *outputFileName)
{
	ScanIOHandles *ioHandles = new ScanIOHandles();
	ioHandles->wsClientPtr = NULL;
	ioHandles->outputFilePtr = NULL;
	ioHandles->ros_topic = ros_topic;
	ioHandles->ros_message_type = ros_message_type;
	ioHandles->config = config;

	// this can throw an exception which will be allowed to propagate since it's fatal
	if (bridgeEndpoint != NULL) ioHandles->wsClientPtr = InitWSConnection(bridgeEndpoint, ros_topic, ros_message_type);
	
	if (outputFileName != NULL) {
		ioHandles->outputFilePtr = new ofstream(outputFileName);
		WriteCaptureStart(*(ioHandles->outputFilePtr));
	}

	ioHandles->deviceHandle = XtsOpenDevice(deviceName);
	if (!ioHandles->deviceHandle) {
		cerr << "Device " << deviceName << " could not be found." << endl;
		isShutdown = TRUE;
	}
	else {
		// Get scan settings, if specfied. Otherwise rely on defaults
		int configVal;
		if (ioHandles->config->lookupValue(XTS_SCAN_MODE_PARAM, configVal)) ioHandles->mode = configVal;
		if (ioHandles->config->lookupValue(XTS_SCAN_FORMAT_PARAM, configVal)) ioHandles->dataFormat = configVal;
		if (ioHandles->config->lookupValue(XTS_SCAN_ACQ_PARAM, configVal)) ioHandles->dataAcq = configVal;
		cout << "mode : " << ioHandles->mode << endl;
		cout << "dataFormat : " << ioHandles->dataFormat << endl;
		cout << "dataAcq : " << ioHandles->dataAcq << endl;
	}

	int val;
	if (!isShutdown && XtsGetControl(ioHandles->deviceHandle, XTS_CTRLID_INTERVAL, XTS_GET_CONTROL_DEF, &val) < 0) {
		cerr << "An error has occurred accessing the device." << endl;
		isShutdown = TRUE;
	}
	else {
		cout << "sample interval " << val << endl;
	}

	// TODO: Consider whether we need to customize the scan area (and other device parameters) through config.
	int xmin, xmax, ymin, ymax;
	if (!isShutdown && XtsGetMaxScanArea(ioHandles->deviceHandle, &xmin, &xmax, &ymin, &ymax) < 0) {
		cerr << "An error has occurred accessing the device." << endl;
		isShutdown = TRUE;
	}
	else {
		cout << "max scan area (" << xmin << "," << ymin << ") x (" << xmax << "," << ymax << ")" << endl;
	}

	// Set timeout
	if (!isShutdown && XtsSetControl(ioHandles->deviceHandle, XTS_CTRLID_SCAN_TIMEOUT, XTS_SET_CONTROL_CUR, 1) < 0) {
		cerr << "Could not set timeout on device." << endl;
		isShutdown = TRUE;
	}

	if (!isShutdown)
	{
		// Note the order of allocation : rows (y) x columns (x)
		ioHandles->scanData.Allocate(ymax - ymin + 1, xmax - xmin + 1);

		// Initiate the scan
		if (XtsStartScan(ioHandles->deviceHandle, 0, XtsCallbackFunction, (void *)ioHandles) < 0) {
			cerr << "Could not initiate scan." << endl;
			isShutdown = TRUE;
		}
	}

	if (isShutdown) {
		CloseHandlesAndDelete(ioHandles);
		SetEvent(writerAckEvent);
		ioHandles = NULL;
	}

	return ioHandles;
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * CloseHandlesAndDelete()
 * Close any open IO Handles, reclaim memory.
 *-----------------------------------------------------------------------------------------------------------------------------*/
void CloseHandlesAndDelete(ScanIOHandles *ioHandles) {

	if (ioHandles != NULL) {
		XtsStopScan(ioHandles->deviceHandle);
		XtsCloseDevice(ioHandles->deviceHandle);
		ioHandles->scanData.Deallocate();
		if (ioHandles->wsClientPtr != NULL) {
			CloseWSConnection(ioHandles->wsClientPtr, ioHandles->ros_topic);
			ioHandles->wsClientPtr = NULL;
		}
		if (ioHandles->outputFilePtr != NULL && ioHandles->outputFilePtr->is_open()) {
			WriteCaptureEnd(*(ioHandles->outputFilePtr));
			ioHandles->outputFilePtr->close();
			ioHandles->outputFilePtr = NULL;
		}
		if (ioHandles->config != NULL) {
			delete ioHandles->config;
		}
		delete ioHandles;
	}
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * InitWSConnection()
 * Open the connection to the ROS bridge server
 *-----------------------------------------------------------------------------------------------------------------------------*/
websocket_client* InitWSConnection(const char *bridgeEndpoint, const char *ros_topic, const char *ros_message_type)
{
	// try connecting to the ROS bridge
	websocket_client* wsClientPtr = new websocket_client();

	// NOTE: The following is windows-specific since web::uri is std::wsstring. Since the Xts API is Windows-specific in itself, this 
	// is acceptable.
	wsClientPtr->connect(wstring(bridgeEndpoint, bridgeEndpoint + strlen(bridgeEndpoint))).wait();
	cout << "Connected to " << bridgeEndpoint << endl;

	// have to advertise the topic and message type
	string advertiseMessage("{ \"op\": \"advertise\", \"topic\": \"");
	advertiseMessage.append(ros_topic);
	advertiseMessage.append("\", \"type\": \"");
	advertiseMessage.append(ros_message_type);
	advertiseMessage.append("\" }");
	PublishMessage(wsClientPtr, advertiseMessage);

	return wsClientPtr;
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * CloseWSConnection()
 * Close the connection to the ROS bridge server
 *-----------------------------------------------------------------------------------------------------------------------------*/
void CloseWSConnection(websocket_client *wsClientPtr, const char *ros_topic)
{
	if (wsClientPtr == NULL) {
		return;
	}

	// unadvertise the topic and message type
	string unadvertiseMessage("{ \"op\": \"unadvertise\", \"topic\": \"");
	unadvertiseMessage.append(ros_topic);
	unadvertiseMessage.append("\" }");
	PublishMessage(wsClientPtr, unadvertiseMessage);

	wsClientPtr->close().wait();
	cout << "Terminated websocket connection" << endl;
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * XtsCallbackFunction()
 * The scan callback.
 *-----------------------------------------------------------------------------------------------------------------------------*/
void __stdcall XtsCallbackFunction(XtsDeviceHandle deviceHandle, int status, DWORD frameNumber, void *arg)
{
	ScanIOHandles *ioHandles = (ScanIOHandles *)arg;

#ifdef DEBUG
	cout << "Callback, status is " << status << endl;
#endif

	if (status < 0)	{
		cout << "XtsCallbackFunction : Device indicates that the thread has completed." << endl;
        // signal shutdown acknowledgement event, just in case
		SetEvent(writerAckEvent);
	} 
	else {
		// if we are already shutdown, don't bother with more frames
		if (!isShutdown) {
			GetDataAndPublish(ioHandles);
		}

		// now check shutdown flag, after completely writing the last frame
		if (isShutdown) {
			// signal that writing the last record has completed, this ensures full frames are recorded
			SetEvent(writerAckEvent);
		}
	}
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * GetDataAndPublish()
 * Get the data from the device and publish it
 *-----------------------------------------------------------------------------------------------------------------------------*/
void GetDataAndPublish(ScanIOHandles *ioHandles)
{	
	DWORD frameNumber;

	 // Get the scan data
	int err = XtsGetScanData(ioHandles->deviceHandle, ioHandles->mode, ioHandles->dataFormat, ioHandles->dataAcq, ioHandles->scanData.ColumnCount() * sizeof(WORD), ioHandles->scanData.RowCount(), &(ioHandles->scanData[0][0]), &frameNumber);

	if (err < 0) {
		cerr << "Device has reported an error." << endl;
		exit(1);
	} 
	else if (err > 0) {
		cerr << "Device has timed out" << endl;
		exit(1);
	} 
	else if (ioHandles->wsClientPtr != NULL) {
		// create message from scanData
		string jsonMessage = MakeScanJSON(frameNumber, ioHandles->ros_topic, ioHandles->scanData);

		// send to ROS bridge
		PublishMessage(ioHandles->wsClientPtr, jsonMessage);
	}
	else if (ioHandles->outputFilePtr != NULL) {
		WriteScanToFile(*(ioHandles->outputFilePtr), frameNumber, ioHandles->scanData);
	}
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * MakeScanJSON()
 * Create the JSON for the ROS bridge publish request.
 *
 * RosBridge publish request JSON:
 *     { "op" : "publish",
 *       "topic" : <topic>,
 *       "msg" : <msg_json> }
 *-----------------------------------------------------------------------------------------------------------------------------*/
string MakeScanJSON(DWORD frameNumber, const char *ros_topic, CMatrix<WORD> &scanData)
{
	json frameJSON;

	frameJSON["frameNumber"] = frameNumber;
	frameJSON["xCoils"] = scanData.ColumnCount();
	frameJSON["yCoils"] = scanData.RowCount();
	frameJSON["frameData"] = json::array();
	for (int row = 0; row < scanData.RowCount(); row++) {
		json nextRowArray = json::array();
		for (int column = 0; column < scanData.ColumnCount(); column++) {
			// The Xts API stuffs data into 8 or 16 bits, depending on the data format setting, convert it to a floating pt
			nextRowArray.push_back((double) scanData[row][column] / 256.0);
		}
		json nextRow;
		nextRow["xCoil"] = nextRowArray;
		frameJSON["frameData"].push_back(nextRow);
	}

	json publishJSON;

	publishJSON["op"] = "publish";
	publishJSON["topic"] = ros_topic;
	publishJSON["msg"] = frameJSON;

	return publishJSON.dump();
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * MakeScanJSON()
 * Create the JSON for the ROS bridge publish request
 * for data from file (CMatrix<FLOAT>).
 *-----------------------------------------------------------------------------------------------------------------------------*/
string MakeScanJSON(DWORD frameNumber, const char *ros_topic, CMatrix<FLOAT> &scanData)
{
	json frameJSON;

	frameJSON["frameNumber"] = frameNumber;
	frameJSON["xCoils"] = scanData.ColumnCount();
	frameJSON["yCoils"] = scanData.RowCount();
	frameJSON["frameData"] = json::array();
	for (int row = 0; row < scanData.RowCount(); row++) {
		json nextRowArray = json::array();
		for (int column = 0; column < scanData.ColumnCount(); column++) {
			nextRowArray.push_back(scanData[row][column]);
		}
		json nextRow;
		nextRow["xCoil"] = nextRowArray;
		frameJSON["frameData"].push_back(nextRow);
	}

	json publishJSON;

	publishJSON["op"] = "publish";
	publishJSON["topic"] = ros_topic;
	publishJSON["msg"] = frameJSON;

	return publishJSON.dump();
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * PublishFromFile()
 * Publish from a sample file - it assumes scan size is 96 columns x 20 rows. As noted in the argument
 * parsing section above, the obscure case of file to file supported in this function for development
 * testing purposes.
 *-----------------------------------------------------------------------------------------------------------------------------*/
void PublishFromFile(const char *fileName, const char *bridgeEndpoint, const char *ros_topic, const char *ros_message_type, const char *outputFileName)
{
	websocket_client *wsClientPtr = NULL;
	ofstream *outputFilePtr = NULL;

	if (bridgeEndpoint != NULL) {
		// this may thrown an exception but is allowed to propagate as fatal
		wsClientPtr = InitWSConnection(bridgeEndpoint, ros_topic, ros_message_type);
	}
	else {
		outputFilePtr = new ofstream(outputFileName);
		WriteCaptureStart(*outputFilePtr);
	}

	string dataLine;
	ifstream dataFile(fileName);
	if (!dataFile.is_open()) {
		cerr << "Could not open file " << fileName << endl;
		return;
	}

	// The # of rows/columns will be read from the first frame of data in the file
	bool allocated = FALSE;
	CMatrix<FLOAT> scanData;
	DWORD frameNumber;

	while (getline(dataFile, dataLine) && !isShutdown)
	{
#ifdef DEBUG
		cout << "dataLine : " << dataLine << endl;
#endif
		if (dataLine.find("Capture") == 0) continue;
		else if (dataLine.find("frameno") == 0)
		{
			// extract frame number
			frameNumber = stoi(dataLine.substr(8));
			if (allocated) 
				// skip next 11 lines if scandata has been allocated
				for (int i = 0; i < 11; i++) getline(dataFile, dataLine);
			else {
				// find coil numbers in order to allocate data
				int xcoilmin = 0, xcoilmax = 0, ycoilmin = 0, ycoilmax = 0;
				for (int i = 0; i < 11; i++) {
					getline(dataFile, dataLine);
					if (dataLine.find("xcoilmin") == 0) xcoilmin = stoi(dataLine.substr(dataLine.find("=") + 1));
					else if (dataLine.find("xcoilmax") == 0) xcoilmax = stoi(dataLine.substr(dataLine.find("=") + 1));
					else if (dataLine.find("ycoilmin") == 0) ycoilmin = stoi(dataLine.substr(dataLine.find("=") + 1));
					else if (dataLine.find("ycoilmax") == 0) ycoilmax = stoi(dataLine.substr(dataLine.find("=") + 1));
				}
				scanData.Allocate(ycoilmax - ycoilmin + 1, xcoilmax - xcoilmin + 1);
				allocated = TRUE;
			}
		}
		else
		{
			// data lines
			for (int row = 0; row < scanData.RowCount(); row++)
			{
				size_t sz = 4; // skip the first column: the row number
				size_t nextNum;
				for (int col = 0; col < scanData.ColumnCount(); col++)
				{
					FLOAT newVal = stof(dataLine.substr(sz), &nextNum);
					sz += nextNum;
#ifdef DEBUG
					cout << "(" << row << ", " << col << ") newVal : " << newVal << ", sz : " << sz << endl;
#endif
					scanData[row][col] = newVal;
				}
				if (row < scanData.RowCount() - 1)
				{
					getline(dataFile, dataLine);
#ifdef DEBUG
					cout << "dataLine : " << dataLine << endl;
#endif
				}
			}

			if (wsClientPtr != NULL) {
				// create message from scanData
				string jsonMessage = MakeScanJSON(frameNumber, ros_topic, scanData);

				// Send message to ROS bridge
				PublishMessage(wsClientPtr, jsonMessage);
			}
			else {
				WriteScanToFile(*outputFilePtr, frameNumber, scanData);
			}
		}
	}

	// done with file or shutdown, clean up what was created in this function
	scanData.Deallocate();
	if (dataFile.is_open()) dataFile.close();
	if (wsClientPtr != NULL) CloseWSConnection(wsClientPtr, ros_topic);
	if (outputFilePtr != NULL) {
		if (outputFilePtr->is_open()) {
			WriteCaptureEnd(*outputFilePtr);
			outputFilePtr->close();
		}
		delete outputFilePtr;
	}
	// acknowledge completion - whether as a result of finishing the file or ctrl-C

}


/*-----------------------------------------------------------------------------------------------------------------------------
 * PublishMessage()
 * Send the JSON message to the web service.
 *-----------------------------------------------------------------------------------------------------------------------------*/
void PublishMessage(websocket_client *wsClientPtr, string jsonMessage) {
#ifdef DEBUG
	cout << "publishing message : \n" << jsonMessage << endl;
#endif

	websocket_outgoing_message pubRequest;
	pubRequest.set_utf8_message(jsonMessage);

	wsClientPtr->send(pubRequest).wait();
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * WriteRecordHeader()
 * Write the frame header to the output stream.
 *-----------------------------------------------------------------------------------------------------------------------------*/
void WriteRecordHeader(ostream &outputStream, DWORD frameNumber, int rowCount, int colCount) {

	outputStream << "frameno=" << frameNumber << endl;

	outputStream << "timestamp=";
	OutputDateTime(outputStream, "%Y/%m/%d %H:%M:%S", TRUE);
	outputStream << endl;

	// TODO: Have to figure out if we need freq, gain & number
	outputStream << "frequency=0.00" << endl;
	outputStream << "gain=0" << endl;
	outputStream << "number=0" << endl;

	outputStream << "xcoilmin=1" << endl;
	outputStream << "xcoilmax=" << colCount << endl;
	outputStream << "ycoilmin=1" << endl;
	outputStream << "ycoilmax=" << rowCount << endl;

	outputStream << setfill(' ') << right;
	outputStream << setw(3) << 1 << '\t' << setw(3) << 1 << endl;
	outputStream << setw(3) << colCount << '\t' << setw(3) << rowCount << endl;

	outputStream << "amp";
	for (int i = 1; i <= colCount; i++) outputStream << "\t" << setw(3) << i;
	outputStream << endl;
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * WriteScanToFile()
 * Write scan data to the output stream - for data directly from the sensor pad (WORD's).
 *-----------------------------------------------------------------------------------------------------------------------------*/
void WriteScanToFile(ostream &outputStream, DWORD frameNumber, CMatrix<WORD> &scanData) {
	int rowCount = scanData.RowCount();    // sensor collective height, x, rows
	int colCount = scanData.ColumnCount(); // sensor width, y, columns
	WriteRecordHeader(outputStream, frameNumber, rowCount, colCount);

	for (int row = 0; row < rowCount; row++) {
		outputStream << setw(3) << row + 1;
		outputStream << fixed;
		for (int col = 0; col < colCount; col++) {
			// the Xts Api stuffs the value into 8 or 16 bits, depending on the data format setting, convert it to a floating point
			outputStream << '\t' << setprecision(4) << setw(10) << (double) scanData[row][col] / 256.0;
		}
		outputStream << endl;
	}
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * WriteCaptureStart()
 * Write timestamped message of capture start.
 *-----------------------------------------------------------------------------------------------------------------------------*/
void WriteCaptureStart(ostream &outputStream) {
	OutputDateTime(outputStream, "Capture started at %a %b %d %H:%M:%S %Y\n"); // "Thu Feb 28 16:27:38 2019"
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * WriteCaptureEnd()
 * Write timestamped message of capture completion.
 *-----------------------------------------------------------------------------------------------------------------------------*/
void WriteCaptureEnd(ostream &outputStream) {
	OutputDateTime(outputStream, "Capture stopped at %a %b %d %H:%M:%S %Y\n");
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * WriteScanToFile()
 * Write scan data to the output stream - for data from file (FLOAT's).
 *-----------------------------------------------------------------------------------------------------------------------------*/
void WriteScanToFile(ostream &outputStream, DWORD frameNumber, CMatrix<FLOAT> &scanData) {
	int rowCount = scanData.RowCount();    // sensor collective height, x, rows
	int colCount = scanData.ColumnCount(); // sensor width, y, columns
	WriteRecordHeader(outputStream, frameNumber, rowCount, colCount);

	for (int row = 0; row < rowCount; row++) {
		outputStream << setw(3) << row + 1;
		outputStream << fixed;
		for (int col = 0; col < colCount; col++) {
			outputStream << '\t' << setprecision(4) << setw(10) << scanData[row][col];
		}
		outputStream << endl;
	}
}


/*-----------------------------------------------------------------------------------------------------------------------------
 * OutputDateTime()
 * Output a timestamp HH:MM:SS.sss (i.e. includes milliseconds) to the given output stream
 * Taken and modified from : https://stackoverflow.com/questions/24686846/get-current-time-in-milliseconds-or-hhmmssmmm-format
 *-----------------------------------------------------------------------------------------------------------------------------*/
void OutputDateTime(ostream &outputStream, const char *formatString, bool appendMillis) {
	// get current time
	auto now = system_clock::now();

	// get number of milliseconds for the current second
	// (remainder after division into seconds)
	auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

	// convert to std::time_t in order to convert to std::tm (broken time)
	auto timer = system_clock::to_time_t(now);

	// convert to broken time
	std::tm bt;
	localtime_s(&bt, &timer);

	outputStream << put_time(&bt, formatString);
	if (appendMillis) 	outputStream << '.' << setfill('0') << setw(3) << ms.count();
}