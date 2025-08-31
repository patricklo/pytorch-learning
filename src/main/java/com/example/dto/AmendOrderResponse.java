package com.example.dto;

import com.example.entity.ClientOrder;

import java.time.LocalDateTime;

public class AmendOrderResponse {

    private boolean success;
    private String message;
    private String lockToken;
    private LocalDateTime lockExpiresAt;
    private ClientOrder order;
    private String errorCode;

    // Constructors
    public AmendOrderResponse() {}

    public AmendOrderResponse(boolean success, String message) {
        this.success = success;
        this.message = message;
    }

    public AmendOrderResponse(boolean success, String message, String lockToken, LocalDateTime lockExpiresAt, ClientOrder order) {
        this.success = success;
        this.message = message;
        this.lockToken = lockToken;
        this.lockExpiresAt = lockExpiresAt;
        this.order = order;
    }

    public AmendOrderResponse(boolean success, String message, String errorCode) {
        this.success = success;
        this.message = message;
        this.errorCode = errorCode;
    }

    // Static factory methods
    public static AmendOrderResponse success(String message, String lockToken, LocalDateTime lockExpiresAt, ClientOrder order) {
        return new AmendOrderResponse(true, message, lockToken, lockExpiresAt, order);
    }

    public static AmendOrderResponse success(String message) {
        return new AmendOrderResponse(true, message);
    }

    public static AmendOrderResponse error(String message) {
        return new AmendOrderResponse(false, message);
    }

    public static AmendOrderResponse error(String message, String errorCode) {
        return new AmendOrderResponse(false, message, errorCode);
    }

    // Getters and Setters
    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getLockToken() {
        return lockToken;
    }

    public void setLockToken(String lockToken) {
        this.lockToken = lockToken;
    }

    public LocalDateTime getLockExpiresAt() {
        return lockExpiresAt;
    }

    public void setLockExpiresAt(LocalDateTime lockExpiresAt) {
        this.lockExpiresAt = lockExpiresAt;
    }

    public ClientOrder getOrder() {
        return order;
    }

    public void setOrder(ClientOrder order) {
        this.order = order;
    }

    public String getErrorCode() {
        return errorCode;
    }

    public void setErrorCode(String errorCode) {
        this.errorCode = errorCode;
    }

    @Override
    public String toString() {
        return "AmendOrderResponse{" +
                "success=" + success +
                ", message='" + message + '\'' +
                ", lockToken='" + lockToken + '\'' +
                ", lockExpiresAt=" + lockExpiresAt +
                ", errorCode='" + errorCode + '\'' +
                '}';
    }
} 