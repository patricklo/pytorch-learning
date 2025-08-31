package com.example.dto;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.math.BigDecimal;

public class AmendOrderRequest {

    @NotNull(message = "Order ID is required")
    private Long orderId;

    @NotBlank(message = "User ID is required")
    @Size(max = 50, message = "User ID must not exceed 50 characters")
    private String userId;

    @Size(max = 20, message = "Order type must not exceed 20 characters")
    private String orderType;

    @Size(max = 20, message = "Status must not exceed 20 characters")
    private String status;

    @DecimalMin(value = "0.01", message = "Amount must be greater than 0")
    private BigDecimal amount;

    @Size(max = 3, message = "Currency must not exceed 3 characters")
    private String currency;

    private String description;

    // Constructors
    public AmendOrderRequest() {}

    public AmendOrderRequest(Long orderId, String userId) {
        this.orderId = orderId;
        this.userId = userId;
    }

    // Getters and Setters
    public Long getOrderId() {
        return orderId;
    }

    public void setOrderId(Long orderId) {
        this.orderId = orderId;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getOrderType() {
        return orderType;
    }

    public void setOrderType(String orderType) {
        this.orderType = orderType;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public BigDecimal getAmount() {
        return amount;
    }

    public void setAmount(BigDecimal amount) {
        this.amount = amount;
    }

    public String getCurrency() {
        return currency;
    }

    public void setCurrency(String currency) {
        this.currency = currency;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    @Override
    public String toString() {
        return "AmendOrderRequest{" +
                "orderId=" + orderId +
                ", userId='" + userId + '\'' +
                ", orderType='" + orderType + '\'' +
                ", status='" + status + '\'' +
                ", amount=" + amount +
                ", currency='" + currency + '\'' +
                '}';
    }
} 