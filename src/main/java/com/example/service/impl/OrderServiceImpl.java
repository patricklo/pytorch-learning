package com.example.service.impl;

import com.example.dto.AmendOrderRequest;
import com.example.dto.AmendOrderResponse;
import com.example.entity.ClientOrder;
import com.example.repository.ClientOrderRepository;
import com.example.service.LockService;
import com.example.service.OrderService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.OptimisticLockingFailureException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Service
public class OrderServiceImpl implements OrderService {

    private static final Logger logger = LoggerFactory.getLogger(OrderServiceImpl.class);

    private final ClientOrderRepository clientOrderRepository;
    private final LockService lockService;

    @Value("${app.lock.timeout:30000}")
    private long defaultLockTimeout;

    public OrderServiceImpl(ClientOrderRepository clientOrderRepository, LockService lockService) {
        this.clientOrderRepository = clientOrderRepository;
        this.lockService = lockService;
    }

    @Override
    @Transactional
    public AmendOrderResponse amendOrder(AmendOrderRequest request) {
        logger.info("Starting amendment process for order {} by user {}", request.getOrderId(), request.getUserId());
        int[][] test = new int[][];
        int rows = test.length;
        int cols = test[0].length;
        // Step 1: Acquire lock
        AmendOrderResponse lockResponse = lockService.acquireLock(
            request.getOrderId(), 
            request.getUserId(), 
            (int) (defaultLockTimeout / 1000)
        );

        if (!lockResponse.isSuccess()) {
            logger.warn("Failed to acquire lock for order {} by user {}: {}", 
                       request.getOrderId(), request.getUserId(), lockResponse.getMessage());
            return lockResponse;
        }

        String lockToken = lockResponse.getLockToken();
        logger.info("Lock acquired for order {} by user {} with token {}", 
                   request.getOrderId(), request.getUserId(), lockToken);

        try {
            // Step 2: Perform the amendment
            return performAmendment(request, lockToken);
        } catch (Exception e) {
            logger.error("Error during amendment for order {} by user {}: {}", 
                        request.getOrderId(), request.getUserId(), e.getMessage(), e);
            
            // Attempt to release lock on error
            try {
                lockService.releaseLock(lockToken, request.getUserId());
                logger.info("Lock released after error for order {} by user {}", 
                           request.getOrderId(), request.getUserId());
            } catch (Exception lockReleaseError) {
                logger.error("Failed to release lock after error for order {} by user {}: {}", 
                           request.getOrderId(), request.getUserId(), lockReleaseError.getMessage());
            }
            
            return AmendOrderResponse.error("Amendment failed: " + e.getMessage(), "AMENDMENT_FAILED");
        }
    }

    @Override
    @Transactional
    public AmendOrderResponse amendOrderWithLock(AmendOrderRequest request, String lockToken) {
        logger.info("Amending order {} by user {} with existing lock token {}", 
                   request.getOrderId(), request.getUserId(), lockToken);

        // Validate the lock
        if (!lockService.validateLock(lockToken, request.getUserId())) {
            logger.warn("Invalid lock token {} for user {} attempting to amend order {}", 
                       lockToken, request.getUserId(), request.getOrderId());
            return AmendOrderResponse.error("Invalid or expired lock token", "INVALID_LOCK");
        }

        try {
            return performAmendment(request, lockToken);
        } catch (Exception e) {
            logger.error("Error during amendment with lock for order {} by user {}: {}", 
                        request.getOrderId(), request.getUserId(), e.getMessage(), e);
            return AmendOrderResponse.error("Amendment failed: " + e.getMessage(), "AMENDMENT_FAILED");
        }
    }

    @Transactional
    protected AmendOrderResponse performAmendment(AmendOrderRequest request, String lockToken) {
        logger.debug("Performing amendment for order {} by user {}", request.getOrderId(), request.getUserId());

        // Get the order with optimistic locking
        Optional<ClientOrder> orderOpt = clientOrderRepository.findByIdWithOptimisticLock(request.getOrderId());
        if (orderOpt.isEmpty()) {
            logger.warn("Order {} not found during amendment", request.getOrderId());
            return AmendOrderResponse.error("Order not found", "ORDER_NOT_FOUND");
        }

        ClientOrder order = orderOpt.get();
        logger.debug("Retrieved order {} with version {} for amendment", order.getId(), order.getVersion());

        // Apply amendments
        boolean hasChanges = false;

        if (request.getOrderType() != null && !request.getOrderType().equals(order.getOrderType())) {
            order.setOrderType(request.getOrderType());
            hasChanges = true;
            logger.debug("Updated order type to {}", request.getOrderType());
        }

        if (request.getStatus() != null && !request.getStatus().equals(order.getStatus())) {
            order.setStatus(request.getStatus());
            hasChanges = true;
            logger.debug("Updated status to {}", request.getStatus());
        }

        if (request.getAmount() != null && !request.getAmount().equals(order.getAmount())) {
            order.setAmount(request.getAmount());
            hasChanges = true;
            logger.debug("Updated amount to {}", request.getAmount());
        }

        if (request.getCurrency() != null && !request.getCurrency().equals(order.getCurrency())) {
            order.setCurrency(request.getCurrency());
            hasChanges = true;
            logger.debug("Updated currency to {}", request.getCurrency());
        }

        if (request.getDescription() != null && !request.getDescription().equals(order.getDescription())) {
            order.setDescription(request.getDescription());
            hasChanges = true;
            logger.debug("Updated description");
        }

        if (!hasChanges) {
            logger.info("No changes detected for order {} by user {}", request.getOrderId(), request.getUserId());
            return AmendOrderResponse.success("No changes to apply", lockToken, 
                                            lockService.getLockExpiration(lockToken), order);
        }

        // Update the updated_by field
        order.setUpdatedBy(request.getUserId());

        try {
            // Save the amended order
            ClientOrder savedOrder = clientOrderRepository.save(order);
            logger.info("Successfully amended order {} by user {} with version {}", 
                       savedOrder.getId(), request.getUserId(), savedOrder.getVersion());

            return AmendOrderResponse.success(
                "Order amended successfully", 
                lockToken, 
                lockService.getLockExpiration(lockToken), 
                savedOrder
            );

        } catch (OptimisticLockingFailureException e) {
            logger.warn("Optimistic locking failure for order {} by user {}: {}", 
                       request.getOrderId(), request.getUserId(), e.getMessage());
            return AmendOrderResponse.error("Order was modified by another user. Please retry.", "CONCURRENT_MODIFICATION");
        } catch (Exception e) {
            logger.error("Error saving amended order {} by user {}: {}", 
                        request.getOrderId(), request.getUserId(), e.getMessage(), e);
            throw new RuntimeException("Failed to save amended order", e);
        }
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<ClientOrder> getOrderById(Long orderId) {
        return clientOrderRepository.findById(orderId);
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<ClientOrder> getOrderByNumber(String orderNumber) {
        return clientOrderRepository.findByOrderNumber(orderNumber);
    }

    @Override
    @Transactional(readOnly = true)
    public List<ClientOrder> getOrdersByClientId(String clientId) {
        return clientOrderRepository.findByClientId(clientId);
    }

    @Override
    @Transactional(readOnly = true)
    public List<ClientOrder> getOrdersByStatus(String status) {
        return clientOrderRepository.findByStatus(status);
    }

    @Override
    @Transactional
    public ClientOrder createOrder(ClientOrder order) {
        logger.info("Creating new order for client {} by user {}", order.getClientId(), order.getCreatedBy());
        
        // Validate order number uniqueness
        if (clientOrderRepository.existsByOrderNumber(order.getOrderNumber())) {
            throw new IllegalArgumentException("Order number already exists: " + order.getOrderNumber());
        }

        ClientOrder savedOrder = clientOrderRepository.save(order);
        logger.info("Successfully created order {} with number {}", savedOrder.getId(), savedOrder.getOrderNumber());
        
        return savedOrder;
    }

    @Override
    @Transactional(readOnly = true)
    public boolean canAmendOrder(Long orderId, String userId) {
        // Check if order exists
        if (!clientOrderRepository.existsById(orderId)) {
            return false;
        }

        // Check if order is locked by another user
        if (lockService.isOrderLocked(orderId)) {
            // Check if the lock belongs to this user
            // This is a simplified check - in a real implementation, you might want to check the specific lock
            return false;
        }

        return true;
    }

    /**
     * Emergency method to force release all locks for an order
     * This should only be used in exceptional circumstances
     */
    @Transactional
    public boolean emergencyReleaseLocks(Long orderId) {
        logger.warn("Emergency release of all locks for order {}", orderId);
        
        try {
            // This would require a method in the lock repository to release all locks for an order
            // For now, we'll use the cleanup method
            int cleanedCount = lockService.cleanupExpiredLocks();
            logger.info("Emergency cleanup completed: {} locks cleaned", cleanedCount);
            return true;
        } catch (Exception e) {
            logger.error("Error during emergency lock release for order {}: {}", orderId, e.getMessage(), e);
            return false;
        }
    }
} 