# operalten/services/ingest-grid-service/messaging.py
# Handles connection and publishing to Kafka.

import logging
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError

# We use a global variable to hold the producer instance,
# ensuring we don't create a new connection for every message.
producer = None

def get_kafka_producer(bootstrap_servers: str):
    """
    Initializes and returns a Kafka producer instance.
    If an instance already exists, it returns the existing one.
    """
    global producer
    if producer is None:
        try:
            logging.info(f"Initializing Kafka producer with bootstrap servers: {bootstrap_servers}")
            producer = KafkaProducer(
                # The list of Kafka brokers to connect to.
                bootstrap_servers=bootstrap_servers.split(','),
                # A function to serialize the message value to JSON bytes.
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                # Number of retries if a send fails.
                retries=5,
                # Wait up to 100ms to batch messages together for efficiency.
                linger_ms=100
            )
        except KafkaError as e:
            logging.error(f"Failed to initialize Kafka producer: {e}")
            producer = None # Ensure it's None on failure so we can retry later.
    return producer

def publish_event(producer_instance: KafkaProducer, topic: str, event: dict):
    """Publishes a single event to a Kafka topic."""
    if not producer_instance:
        logging.error("Kafka producer is not available. Cannot publish event.")
        return

    try:
        event_id = event.get('eventId', 'N/A')
        logging.info(f"Publishing event to topic '{topic}': {event_id}")

        # The key helps in partitioning data. For example, all data from the same
        # ISO can be sent to the same partition, ensuring ordered processing if needed.
        key = event.get('iso', 'UNKNOWN').encode('utf-8')

        # Asynchronously send the message.
        future = producer_instance.send(topic, key=key, value=event)

        # You can optionally block and wait for the result for debugging or critical messages.
        # record_metadata = future.get(timeout=10)
        # logging.info(f"Successfully published event. Topic: {record_metadata.topic}, Partition: {record_metadata.partition}")

    except KafkaError as e:
        logging.error(f"Failed to publish event to Kafka: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during Kafka publishing: {e}")