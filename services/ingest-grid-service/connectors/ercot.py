# operalten/services/ingest-grid-service/connectors/ercot.py
# Contains the logic for fetching and processing data from ERCOT.

import logging
import requests
import uuid
from datetime import datetime, timezone

def _transform_data(raw_data: dict, api_url: str) -> dict:
    """Transforms raw ERCOT API data into our standard event schema."""
    # The raw data from ERCOT's dashboard API is nested.
    # We navigate to the 'current_condition' section to get the load value.
    current_load = raw_data.get("current_condition", {}).get("actual_load")

    if not current_load:
        logging.warning("No 'actual_load' found in raw ERCOT data.")
        return None

    # This matches the event schema we designed earlier
    transformed_event = {
        "eventId": str(uuid.uuid4()),
        "source": "ERCOT_API_LOAD_DASH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "iso": "ERCOT",
        "market_type": "REAL_TIME",
        "data_points": [
            {
                "type": "LOAD_MW", # ERCOT reports in Megawatts
                "value": current_load,
                "region": "SystemWide"
            }
        ],
        "data_provenance": f"Source: {api_url} @ {datetime.now(timezone.utc).isoformat()}"
    }
    return transformed_event

def poll_ercot(api_url: str):
    """Polls the ERCOT API and returns a standardized data event."""
    logging.info(f"Polling ERCOT endpoint: {api_url}")
    try:
        # Some APIs require specific headers to identify the client
        headers = {'User-Agent': 'OperAlten-Data-Ingestor/1.0'}
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status() # This will raise an error for bad responses (4xx or 5xx)
        
        raw_data = response.json()
        logging.info("Successfully fetched raw data from ERCOT.")
        
        # Transform the data into our standard format
        standardized_data = _transform_data(raw_data, api_url)
        if standardized_data:
            logging.info("ERCOT data transformed successfully.")
            return standardized_data
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from ERCOT: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing ERCOT data: {e}")
        
    return None