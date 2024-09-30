# utils/sqlalchemy_events.py

from sqlalchemy import event
from models.database import User, Agent, Customer
import logging

logger = logging.getLogger(__name__)

def create_agent_entry(mapper, connection, target):
    if target.role == 'agent':
        new_agent = Agent(user_id=target.id)
        connection.execute(
            Agent.__table__.insert(),
            {"user_id": new_agent.user_id}
        )
        logger.info(f"Agent entry created for user_id: {target.id}")

def delete_agent_entry(mapper, connection, target):
    if target.role == 'agent':
        connection.execute(
            Agent.__table__.delete().where(Agent.user_id == target.id)
        )
        logger.info(f"Agent entry deleted for user_id: {target.id}")

def create_customer_entry(mapper, connection, target):
    if target.role == 'user':
        new_customer = Customer(user_id=target.id)
        connection.execute(
            Customer.__table__.insert(),
            {"user_id": new_customer.user_id}
        )
        logger.info(f"Customer entry created for user_id: {target.id}")

def delete_customer_entry(mapper, connection, target):
    if target.role == 'user':
        connection.execute(
            Customer.__table__.delete().where(Customer.user_id == target.id)
        )
        logger.info(f"Customer entry deleted for user_id: {target.id}")

def register_listeners():
    """
    Registers SQLAlchemy event listeners.
    """
    event.listen(User, 'after_insert', create_agent_entry)
    event.listen(User, 'after_delete', delete_agent_entry)
    event.listen(User, 'after_insert', create_customer_entry)
    event.listen(User, 'after_delete', delete_customer_entry)
    logger.info("SQLAlchemy event listeners registered.")
