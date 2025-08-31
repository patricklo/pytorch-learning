package com.example.controller;

import com.example.dto.AmendOrderRequest;
import com.example.dto.AmendOrderResponse;
import com.example.entity.ClientOrder;
import com.example.service.LockService;
import com.example.service.OrderService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/orders")
public class OrderController {

    private static final Logger logger = LoggerFactory.getLogger(OrderController.class);

    private final OrderService orderService;
    private final LockService lockService;

    public OrderController(OrderService orderService, LockService lockService) {
        this.orderService = orderService;
        this.lockService = lockService;
    }

    /**
     * Amend a ClientOrder with exclusive locking
     * POST /api/orders/amend
     */
    @PostMapping("/amend")
    public ResponseEntity<AmendOrderResponse> amendOrder(@Valid @RequestBody AmendOrderRequest request) {
        logger.info("Received amend request for order {} by user {}", request.getOrderId(), request.getUserId());

        try {
            AmendOrderResponse response = orderService.amendOrder(request);
            
            if (response.isSuccess()) {
                logger.info("Successfully amended order {} by user {}", request.getOrderId(), request.getUserId());
                return ResponseEntity.ok(response);
            } else {
                logger.warn("Failed to amend order {} by user {}: {}", 
                           request.getOrderId(), request.getUserId(), response.getMessage());
                
                // Return appropriate HTTP status based on error
                HttpStatus status = getHttpStatusForError(response.getErrorCode());
                return ResponseEntity.status(status).body(response);
            }
        } catch (Exception e) {
            logger.error("Unexpected error during amendment for order {} by user {}: {}", 
                        request.getOrderId(), request.getUserId(), e.getMessage(), e);
            
            AmendOrderResponse errorResponse = AmendOrderResponse.error(
                "Internal server error during amendment", "INTERNAL_ERROR"
            );
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    /**
     * Amend a ClientOrder using an existing lock token
     * POST /api/orders/amend-with-lock
     */
    @PostMapping("/amend-with-lock")
    public ResponseEntity<AmendOrderResponse> amendOrderWithLock(
            @Valid @RequestBody AmendOrderRequest request,
            @RequestHeader("X-Lock-Token") String lockToken) {
        
        logger.info("Received amend request with lock for order {} by user {} with token {}", 
                   request.getOrderId(), request.getUserId(), lockToken);

        try {
            AmendOrderResponse response = orderService.amendOrderWithLock(request, lockToken);
            
            if (response.isSuccess()) {
                logger.info("Successfully amended order {} by user {} with lock token {}", 
                           request.getOrderId(), request.getUserId(), lockToken);
                return ResponseEntity.ok(response);
            } else {
                logger.warn("Failed to amend order {} by user {} with lock token {}: {}", 
                           request.getOrderId(), request.getUserId(), lockToken, response.getMessage());
                
                HttpStatus status = getHttpStatusForError(response.getErrorCode());
                return ResponseEntity.status(status).body(response);
            }
        } catch (Exception e) {
            logger.error("Unexpected error during amendment with lock for order {} by user {}: {}", 
                        request.getOrderId(), request.getUserId(), e.getMessage(), e);
            
            AmendOrderResponse errorResponse = AmendOrderResponse.error(
                "Internal server error during amendment", "INTERNAL_ERROR"
            );
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    /**
     * Release a lock
     * DELETE /api/orders/locks/{lockToken}
     */
    @DeleteMapping("/locks/{lockToken}")
    public ResponseEntity<AmendOrderResponse> releaseLock(
            @PathVariable String lockToken,
            @RequestParam String userId) {
        
        logger.info("Received lock release request for token {} by user {}", lockToken, userId);

        try {
            boolean success = lockService.releaseLock(lockToken, userId);
            
            if (success) {
                logger.info("Successfully released lock {} by user {}", lockToken, userId);
                AmendOrderResponse response = AmendOrderResponse.success("Lock released successfully");
                return ResponseEntity.ok(response);
            } else {
                logger.warn("Failed to release lock {} by user {}", lockToken, userId);
                AmendOrderResponse response = AmendOrderResponse.error("Failed to release lock", "LOCK_RELEASE_FAILED");
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            }
        } catch (Exception e) {
            logger.error("Unexpected error during lock release for token {} by user {}: {}", 
                        lockToken, userId, e.getMessage(), e);
            
            AmendOrderResponse errorResponse = AmendOrderResponse.error(
                "Internal server error during lock release", "INTERNAL_ERROR"
            );
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    /**
     * Extend a lock
     * PUT /api/orders/locks/{lockToken}/extend
     */
    @PutMapping("/locks/{lockToken}/extend")
    public ResponseEntity<AmendOrderResponse> extendLock(
            @PathVariable String lockToken,
            @RequestParam String userId,
            @RequestParam(defaultValue = "300") int additionalSeconds) {
        
        logger.info("Received lock extension request for token {} by user {} for {} seconds", 
                   lockToken, userId, additionalSeconds);

        try {
            boolean success = lockService.extendLock(lockToken, userId, additionalSeconds);
            
            if (success) {
                logger.info("Successfully extended lock {} by user {} for {} seconds", 
                           lockToken, userId, additionalSeconds);
                AmendOrderResponse response = AmendOrderResponse.success("Lock extended successfully");
                return ResponseEntity.ok(response);
            } else {
                logger.warn("Failed to extend lock {} by user {}", lockToken, userId);
                AmendOrderResponse response = AmendOrderResponse.error("Failed to extend lock", "LOCK_EXTENSION_FAILED");
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            }
        } catch (Exception e) {
            logger.error("Unexpected error during lock extension for token {} by user {}: {}", 
                        lockToken, userId, e.getMessage(), e);
            
            AmendOrderResponse errorResponse = AmendOrderResponse.error(
                "Internal server error during lock extension", "INTERNAL_ERROR"
            );
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    /**
     * Get order by ID
     * GET /api/orders/{orderId}
     */
    @GetMapping("/{orderId}")
    public ResponseEntity<ClientOrder> getOrderById(@PathVariable Long orderId) {
        logger.debug("Received request to get order {}", orderId);

        Optional<ClientOrder> order = orderService.getOrderById(orderId);
        
        if (order.isPresent()) {
            return ResponseEntity.ok(order.get());
        } else {
            return ResponseEntity.notFound().build();
        }
    }

    /**
     * Get order by order number
     * GET /api/orders/number/{orderNumber}
     */
    @GetMapping("/number/{orderNumber}")
    public ResponseEntity<ClientOrder> getOrderByNumber(@PathVariable String orderNumber) {
        logger.debug("Received request to get order by number {}", orderNumber);

        Optional<ClientOrder> order = orderService.getOrderByNumber(orderNumber);
        
        if (order.isPresent()) {
            return ResponseEntity.ok(order.get());
        } else {
            return ResponseEntity.notFound().build();
        }
    }

    /**
     * Get orders by client ID
     * GET /api/orders/client/{clientId}
     */
    @GetMapping("/client/{clientId}")
    public ResponseEntity<List<ClientOrder>> getOrdersByClientId(@PathVariable String clientId) {
        logger.debug("Received request to get orders for client {}", clientId);

        List<ClientOrder> orders = orderService.getOrdersByClientId(clientId);
        return ResponseEntity.ok(orders);
    }

    /**
     * Get orders by status
     * GET /api/orders/status/{status}
     */
    @GetMapping("/status/{status}")
    public ResponseEntity<List<ClientOrder>> getOrdersByStatus(@PathVariable String status) {
        logger.debug("Received request to get orders with status {}", status);

        List<ClientOrder> orders = orderService.getOrdersByStatus(status);
        return ResponseEntity.ok(orders);
    }

    /**
     * Check if order can be amended
     * GET /api/orders/{orderId}/can-amend
     */
    @GetMapping("/{orderId}/can-amend")
    public ResponseEntity<AmendOrderResponse> canAmendOrder(
            @PathVariable Long orderId,
            @RequestParam String userId) {
        
        logger.debug("Received request to check if order {} can be amended by user {}", orderId, userId);

        boolean canAmend = orderService.canAmendOrder(orderId, userId);
        
        if (canAmend) {
            AmendOrderResponse response = AmendOrderResponse.success("Order can be amended");
            return ResponseEntity.ok(response);
        } else {
            AmendOrderResponse response = AmendOrderResponse.error("Order cannot be amended", "ORDER_CANNOT_BE_AMENDED");
            return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
        }
    }

    /**
     * Emergency endpoint to clean up expired locks
     * POST /api/orders/emergency-cleanup
     */
    @PostMapping("/emergency-cleanup")
    public ResponseEntity<AmendOrderResponse> emergencyCleanup() {
        logger.warn("Received emergency cleanup request");

        try {
            int cleanedCount = lockService.cleanupExpiredLocks();
            AmendOrderResponse response = AmendOrderResponse.success(
                String.format("Emergency cleanup completed: %d locks cleaned", cleanedCount)
            );
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error during emergency cleanup: {}", e.getMessage(), e);
            AmendOrderResponse errorResponse = AmendOrderResponse.error(
                "Error during emergency cleanup: " + e.getMessage(), "CLEANUP_ERROR"
            );
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    /**
     * Get lock expiration time
     * GET /api/orders/locks/{lockToken}/expiration
     */
    @GetMapping("/locks/{lockToken}/expiration")
    public ResponseEntity<AmendOrderResponse> getLockExpiration(@PathVariable String lockToken) {
        logger.debug("Received request to get expiration for lock {}", lockToken);

        try {
            var expiration = lockService.getLockExpiration(lockToken);
            
            if (expiration != null) {
                AmendOrderResponse response = AmendOrderResponse.success("Lock expiration retrieved");
                response.setLockExpiresAt(expiration);
                return ResponseEntity.ok(response);
            } else {
                AmendOrderResponse response = AmendOrderResponse.error("Lock not found", "LOCK_NOT_FOUND");
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(response);
            }
        } catch (Exception e) {
            logger.error("Error getting lock expiration for token {}: {}", lockToken, e.getMessage(), e);
            AmendOrderResponse errorResponse = AmendOrderResponse.error(
                "Error getting lock expiration", "INTERNAL_ERROR"
            );
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    private HttpStatus getHttpStatusForError(String errorCode) {
        if (errorCode == null) {
            return HttpStatus.BAD_REQUEST;
        }

        return switch (errorCode) {
            case "ORDER_NOT_FOUND" -> HttpStatus.NOT_FOUND;
            case "ORDER_ALREADY_LOCKED" -> HttpStatus.CONFLICT;
            case "INVALID_LOCK" -> HttpStatus.UNAUTHORIZED;
            case "CONCURRENT_MODIFICATION" -> HttpStatus.CONFLICT;
            case "LOCK_ACQUISITION_FAILED" -> HttpStatus.CONFLICT;
            case "AMENDMENT_FAILED" -> HttpStatus.BAD_REQUEST;
            case "LOCK_RELEASE_FAILED" -> HttpStatus.BAD_REQUEST;
            case "LOCK_EXTENSION_FAILED" -> HttpStatus.BAD_REQUEST;
            case "ORDER_CANNOT_BE_AMENDED" -> HttpStatus.CONFLICT;
            default -> HttpStatus.BAD_REQUEST;
        };
    }
} 