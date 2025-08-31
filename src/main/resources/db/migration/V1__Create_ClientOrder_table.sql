-- Create ClientOrder table
CREATE TABLE client_order (
    id BIGSERIAL PRIMARY KEY,
    order_number VARCHAR(50) NOT NULL UNIQUE,
    client_id VARCHAR(50) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50) NOT NULL,
    updated_by VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);

-- Create indexes for better performance
CREATE INDEX idx_client_order_order_number ON client_order(order_number);
CREATE INDEX idx_client_order_client_id ON client_order(client_id);
CREATE INDEX idx_client_order_status ON client_order(status);
CREATE INDEX idx_client_order_created_at ON client_order(created_at);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_client_order_updated_at 
    BEFORE UPDATE ON client_order 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create table for tracking locks
CREATE TABLE client_order_locks (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT NOT NULL REFERENCES client_order(id) ON DELETE CASCADE,
    locked_by VARCHAR(50) NOT NULL,
    locked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    lock_token VARCHAR(100) NOT NULL UNIQUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Create indexes for lock table
CREATE INDEX idx_client_order_locks_order_id ON client_order_locks(order_id);
CREATE INDEX idx_client_order_locks_expires_at ON client_order_locks(expires_at);
CREATE INDEX idx_client_order_locks_is_active ON client_order_locks(is_active);

-- Create a function to clean up expired locks
CREATE OR REPLACE FUNCTION cleanup_expired_locks()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM client_order_locks 
    WHERE expires_at < CURRENT_TIMESTAMP AND is_active = TRUE;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Create a scheduled job to clean up expired locks (if using pg_cron extension)
-- SELECT cron.schedule('cleanup-expired-locks', '*/5 * * * *', 'SELECT cleanup_expired_locks();');

-- Insert some sample data
INSERT INTO client_order (order_number, client_id, order_type, status, amount, currency, description, created_by, updated_by) VALUES
('ORD-001-2024', 'CLIENT-001', 'BUY', 'PENDING', 1000.00, 'USD', 'Sample order 1', 'system', 'system'),
('ORD-002-2024', 'CLIENT-002', 'SELL', 'PENDING', 2500.50, 'USD', 'Sample order 2', 'system', 'system'),
('ORD-003-2024', 'CLIENT-001', 'BUY', 'PENDING', 750.25, 'USD', 'Sample order 3', 'system', 'system'); 