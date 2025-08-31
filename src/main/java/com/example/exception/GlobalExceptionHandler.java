package com.example.exception;

import com.example.dto.AmendOrderResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.HashMap;
import java.util.Map;

@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger logger = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<AmendOrderResponse> handleValidationExceptions(MethodArgumentNotValidException ex) {
        logger.warn("Validation error: {}", ex.getMessage());

        Map<String, String> errors = new HashMap<>();
        ex.getBindingResult().getAllErrors().forEach((error) -> {
            String fieldName = ((FieldError) error).getField();
            String errorMessage = error.getDefaultMessage();
            errors.put(fieldName, errorMessage);
        });

        String errorMessage = "Validation failed: " + errors.toString();
        AmendOrderResponse response = AmendOrderResponse.error(errorMessage, "VALIDATION_ERROR");
        
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<AmendOrderResponse> handleIllegalArgumentException(IllegalArgumentException ex) {
        logger.warn("Illegal argument error: {}", ex.getMessage());
        
        AmendOrderResponse response = AmendOrderResponse.error(ex.getMessage(), "INVALID_ARGUMENT");
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
    }

    @ExceptionHandler(OrderNotFoundException.class)
    public ResponseEntity<AmendOrderResponse> handleOrderNotFoundException(OrderNotFoundException ex) {
        logger.warn("Order not found: {}", ex.getMessage());
        
        AmendOrderResponse response = AmendOrderResponse.error(ex.getMessage(), "ORDER_NOT_FOUND");
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(response);
    }

    @ExceptionHandler(OrderLockedException.class)
    public ResponseEntity<AmendOrderResponse> handleOrderLockedException(OrderLockedException ex) {
        logger.warn("Order locked error: {}", ex.getMessage());
        
        AmendOrderResponse response = AmendOrderResponse.error(ex.getMessage(), "ORDER_ALREADY_LOCKED");
        return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
    }

    @ExceptionHandler(InvalidLockException.class)
    public ResponseEntity<AmendOrderResponse> handleInvalidLockException(InvalidLockException ex) {
        logger.warn("Invalid lock error: {}", ex.getMessage());
        
        AmendOrderResponse response = AmendOrderResponse.error(ex.getMessage(), "INVALID_LOCK");
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(response);
    }

    @ExceptionHandler(ConcurrentModificationException.class)
    public ResponseEntity<AmendOrderResponse> handleConcurrentModificationException(ConcurrentModificationException ex) {
        logger.warn("Concurrent modification error: {}", ex.getMessage());
        
        AmendOrderResponse response = AmendOrderResponse.error(ex.getMessage(), "CONCURRENT_MODIFICATION");
        return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<AmendOrderResponse> handleGenericException(Exception ex) {
        logger.error("Unexpected error: {}", ex.getMessage(), ex);
        
        AmendOrderResponse response = AmendOrderResponse.error("Internal server error", "INTERNAL_ERROR");
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
    }

    // Custom exception classes
    public static class OrderNotFoundException extends RuntimeException {
        public OrderNotFoundException(String message) {
            super(message);
        }
    }

    public static class OrderLockedException extends RuntimeException {
        public OrderLockedException(String message) {
            super(message);
        }
    }

    public static class InvalidLockException extends RuntimeException {
        public InvalidLockException(String message) {
            super(message);
        }
    }

    public static class ConcurrentModificationException extends RuntimeException {
        public ConcurrentModificationException(String message) {
            super(message);
        }
    }
} 