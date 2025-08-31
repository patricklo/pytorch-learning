package com.example.service.impl;

import com.example.dto.AmendOrderResponse;
import com.example.entity.ClientOrder;
import com.example.entity.ClientOrderLock;
import com.example.repository.ClientOrderLockRepository;
import com.example.repository.ClientOrderRepository;
import com.example.service.LockService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.UUID;

@Service
public class LockServiceImpl implements LockService {

    private static final Logger logger = LoggerFactory.getLogger(LockServiceImpl.class);

    private final ClientOrderRepository clientOrderRepository;
    private final ClientOrderLockRepository lockRepository;

    @Value("${app.lock.timeout:30000}")
    private long defaultLockTimeout;

    @Value("${app.lock.retry-interval:1000}")
    private long retryInterval;

    @Value("${app.lock.max-retries:3}")
    private int maxRetries;

    public LockServiceImpl(ClientOrderRepository clientOrderRepository, 
                          ClientOrderLockRepository lockRepository) {
        this.clientOrderRepository = clientOrderRepository;
        this.lockRepository = lockRepository;
    }

    @Override
    @Transactional
    public AmendOrderResponse acquireLock(Long orderId, String userId, int timeoutSeconds) {
        logger.info("Attempting to acquire lock for order {} by user {}", orderId, userId);

        // First, check if the order exists
        Optional<ClientOrder> orderOpt = clientOrderRepository.findById(orderId);
        if (orderOpt.isEmpty()) {
            logger.warn("Order {} not found", orderId);
            return AmendOrderResponse.error("Order not found", "ORDER_NOT_FOUND");
        }

        ClientOrder order = orderOpt.get();

        // Check if order is already locked
        Optional<ClientOrderLock> existingLock = lockRepository.findActiveLockByOrderId(orderId, LocalDateTime.now());
        if (existingLock.isPresent()) {
            ClientOrderLock lock = existingLock.get();
            logger.warn("Order {} is already locked by user {} until {}", 
                       orderId, lock.getLockedBy(), lock.getExpiresAt());
            
            return AmendOrderResponse.error(
                String.format("Order is currently locked by user %s until %s", 
                             lock.getLockedBy(), lock.getExpiresAt()),
                "ORDER_ALREADY_LOCKED"
            );
        }

        // Clean up any expired locks for this order
        lockRepository.releaseAllLocksForOrder(orderId);

        // Generate lock token
        String lockToken = generateLockToken();
        LocalDateTime expiresAt = LocalDateTime.now().plusSeconds(timeoutSeconds);

        // Create and save the lock
        ClientOrderLock lock = new ClientOrderLock(orderId, userId, expiresAt, lockToken);
        
        try {
            lockRepository.save(lock);
            logger.info("Successfully acquired lock for order {} by user {} with token {}", 
                       orderId, userId, lockToken);
            
            return AmendOrderResponse.success(
                "Lock acquired successfully", 
                lockToken, 
                expiresAt, 
                order
            );
        } catch (DataIntegrityViolationException e) {
            logger.error("Failed to acquire lock for order {} by user {}: {}", orderId, userId, e.getMessage());
            return AmendOrderResponse.error("Failed to acquire lock - concurrent lock attempt", "LOCK_ACQUISITION_FAILED");
        } catch (Exception e) {
            logger.error("Unexpected error acquiring lock for order {} by user {}: {}", orderId, userId, e.getMessage());
            return AmendOrderResponse.error("Internal server error during lock acquisition", "INTERNAL_ERROR");
        }
    }

    @Override
    @Transactional
    public boolean releaseLock(String lockToken, String userId) {
        logger.info("Attempting to release lock with token {} by user {}", lockToken, userId);

        Optional<ClientOrderLock> lockOpt = lockRepository.findActiveLockByToken(lockToken);
        if (lockOpt.isEmpty()) {
            logger.warn("Lock with token {} not found or already released", lockToken);
            return false;
        }

        ClientOrderLock lock = lockOpt.get();
        
        // Check if the lock belongs to the user
        if (!lock.getLockedBy().equals(userId)) {
            logger.warn("User {} attempted to release lock owned by user {}", userId, lock.getLockedBy());
            return false;
        }

        // Check if lock has expired
        if (lock.isExpired()) {
            logger.warn("Lock with token {} has already expired", lockToken);
            // Still mark it as inactive for cleanup
            lockRepository.releaseLockByToken(lockToken);
            return true;
        }

        // Release the lock
        int updatedRows = lockRepository.releaseLockByToken(lockToken);
        boolean success = updatedRows > 0;
        
        if (success) {
            logger.info("Successfully released lock with token {} by user {}", lockToken, userId);
        } else {
            logger.warn("Failed to release lock with token {} by user {}", lockToken, userId);
        }
        
        return success;
    }

    @Override
    @Transactional(readOnly = true)
    public boolean validateLock(String lockToken, String userId) {
        logger.debug("Validating lock with token {} for user {}", lockToken, userId);

        Optional<ClientOrderLock> lockOpt = lockRepository.findActiveLockByToken(lockToken);
        if (lockOpt.isEmpty()) {
            logger.debug("Lock with token {} not found", lockToken);
            return false;
        }

        ClientOrderLock lock = lockOpt.get();
        
        // Check if lock belongs to user and is not expired
        boolean isValid = lock.getLockedBy().equals(userId) && !lock.isExpired();
        
        if (!isValid) {
            logger.debug("Lock with token {} is invalid for user {} (expired: {}, owner: {})", 
                        lockToken, userId, lock.isExpired(), lock.getLockedBy());
        }
        
        return isValid;
    }

    @Override
    @Transactional
    public boolean extendLock(String lockToken, String userId, int additionalSeconds) {
        logger.info("Attempting to extend lock with token {} by user {} for {} seconds", 
                   lockToken, userId, additionalSeconds);

        Optional<ClientOrderLock> lockOpt = lockRepository.findActiveLockByToken(lockToken);
        if (lockOpt.isEmpty()) {
            logger.warn("Lock with token {} not found", lockToken);
            return false;
        }

        ClientOrderLock lock = lockOpt.get();
        
        // Check if the lock belongs to the user
        if (!lock.getLockedBy().equals(userId)) {
            logger.warn("User {} attempted to extend lock owned by user {}", userId, lock.getLockedBy());
            return false;
        }

        // Check if lock has expired
        if (lock.isExpired()) {
            logger.warn("Lock with token {} has already expired", lockToken);
            return false;
        }

        // Extend the lock
        lock.setExpiresAt(lock.getExpiresAt().plusSeconds(additionalSeconds));
        lockRepository.save(lock);
        
        logger.info("Successfully extended lock with token {} by user {} until {}", 
                   lockToken, userId, lock.getExpiresAt());
        return true;
    }

    @Override
    @Transactional
    public int cleanupExpiredLocks() {
        logger.debug("Starting cleanup of expired locks");
        
        int cleanedCount = lockRepository.cleanupExpiredLocks();
        
        if (cleanedCount > 0) {
            logger.info("Cleaned up {} expired locks", cleanedCount);
        } else {
            logger.debug("No expired locks found to clean up");
        }
        
        return cleanedCount;
    }

    @Override
    @Transactional(readOnly = true)
    public LocalDateTime getLockExpiration(String lockToken) {
        Optional<ClientOrderLock> lockOpt = lockRepository.findActiveLockByToken(lockToken);
        return lockOpt.map(ClientOrderLock::getExpiresAt).orElse(null);
    }

    @Override
    @Transactional(readOnly = true)
    public boolean isOrderLocked(Long orderId) {
        return lockRepository.countActiveLocksForOrder(orderId, LocalDateTime.now()) > 0;
    }

    /**
     * Scheduled task to clean up expired locks every 5 minutes
     */
    @Scheduled(fixedRate = 300000) // 5 minutes
    public void scheduledCleanup() {
        try {
            cleanupExpiredLocks();
        } catch (Exception e) {
            logger.error("Error during scheduled lock cleanup: {}", e.getMessage(), e);
        }
    }

    /**
     * Emergency cleanup method that can be called manually
     */
    @Transactional
    public int emergencyCleanup() {
        logger.warn("Performing emergency cleanup of expired locks");
        
        LocalDateTime now = LocalDateTime.now();
        int cleanedCount = 0;
        
        try {
            // Get all expired locks
            var expiredLocks = lockRepository.findAllExpiredLocks(now);
            
            for (ClientOrderLock lock : expiredLocks) {
                try {
                    lock.setIsActive(false);
                    lockRepository.save(lock);
                    cleanedCount++;
                    logger.debug("Emergency cleanup: released expired lock {} for order {}", 
                               lock.getLockToken(), lock.getOrderId());
                } catch (Exception e) {
                    logger.error("Error during emergency cleanup of lock {}: {}", 
                               lock.getLockToken(), e.getMessage());
                }
            }
            
            logger.info("Emergency cleanup completed: {} locks cleaned", cleanedCount);
        } catch (Exception e) {
            logger.error("Error during emergency cleanup: {}", e.getMessage(), e);
        }
        
        return cleanedCount;
    }

    private String generateLockToken() {
        return UUID.randomUUID().toString();
    }
} 