package com.example.service;

import com.example.dto.AmendOrderResponse;
import com.example.entity.ClientOrder;

import java.time.LocalDateTime;

public interface LockService {

    /**
     * Attempts to acquire an exclusive lock on a ClientOrder for amendment
     * @param orderId The ID of the order to lock
     * @param userId The user attempting to acquire the lock
     * @param timeoutSeconds The lock timeout in seconds
     * @return AmendOrderResponse with lock status and token
     */
    AmendOrderResponse acquireLock(Long orderId, String userId, int timeoutSeconds);

    /**
     * Releases a lock using the lock token
     * @param lockToken The token of the lock to release
     * @param userId The user attempting to release the lock
     * @return true if lock was successfully released, false otherwise
     */
    boolean releaseLock(String lockToken, String userId);

    /**
     * Validates if a lock is still valid and belongs to the user
     * @param lockToken The lock token to validate
     * @param userId The user to validate against
     * @return true if lock is valid and belongs to user, false otherwise
     */
    boolean validateLock(String lockToken, String userId);

    /**
     * Extends the expiration time of an existing lock
     * @param lockToken The lock token to extend
     * @param userId The user requesting the extension
     * @param additionalSeconds Additional seconds to add to the lock
     * @return true if lock was successfully extended, false otherwise
     */
    boolean extendLock(String lockToken, String userId, int additionalSeconds);

    /**
     * Cleans up expired locks
     * @return Number of locks cleaned up
     */
    int cleanupExpiredLocks();

    /**
     * Gets the expiration time of a lock
     * @param lockToken The lock token
     * @return The expiration time, or null if lock doesn't exist
     */
    LocalDateTime getLockExpiration(String lockToken);

    /**
     * Checks if an order is currently locked
     * @param orderId The order ID to check
     * @return true if order is locked, false otherwise
     */
    boolean isOrderLocked(Long orderId);
} 